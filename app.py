import time
import logging
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import json
import requests
import os
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pydub import AudioSegment
import re
import subprocess
import streamlit as st
import platform
import sys
from urllib.parse import quote_plus, urlencode

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_FILE_PATH = 'config.json'

def load_config():
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)
    else:
        return {"api_id": "", "affiliate_id": ""}

def save_config(api_id, affiliate_id):
    config = {"api_id": api_id, "affiliate_id": affiliate_id}
    with open(CONFIG_FILE_PATH, 'w') as f:
        json.dump(config, f)

config = load_config()

def fix_video_file(input_path):
    output_path = input_path.replace(".mp4", "_fixed.mp4")
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c', 'copy',
        '-movflags', 'faststart',
        output_path
    ]
    try:
        subprocess.run(command, check=True)
        logger.info(f"動画ファイルを修正しました: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"動画ファイルの修正に失敗しました: {e}")
        st.error("動画ファイルの修正に失敗しました。")
        return None

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--start-maximized")
    chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        return driver
    except Exception as e:
        logger.error(f"ブラウザの設定中にエラーが発生しました: {e}")
        st.error("ブラウザの設定中にエラーが発生しました。Chromeがインストールされているか確認してください。")
        sys.exit(1)

def clear_age_verification(driver):
    try:
        age_verification_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//a[@class="ageCheck__link ageCheck__link--r18"]'))
        )
        age_verification_button.click()
        logger.info("年齢確認をクリアしました。")
    except Exception as e:
        logger.error(f"年齢確認をクリアする中でエラーが発生しました: {e}")
        st.error("年齢確認をクリアする中でエラーが発生しました。")

def close_campaign_popup(driver):
    try:
        close_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'campaign-popup-close'))
        )
        close_button.click()
        logger.info("キャンペーンポップアップを閉じました。")
    except Exception as e:
        logger.error(f"キャンペーンポップアップを閉じる中でエラーが発生しました: {e}")
        st.error("キャンペーンポップアップを閉じる中でエラーが発生しました。")

def fetch_video_page(driver, url):
    driver.get(url)
    clear_age_verification(driver)
    close_campaign_popup(driver)
    logger.info("ページのHTMLを取得しました。")
    html_content = driver.page_source

    try:
        title_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'title'))
        )
        title = title_element.get_attribute('innerHTML').strip()
        title = re.sub(r'[\\/*?:"<>|]', "", title)  # 無効なファイル名文字を除去
        logger.info(f"タイトル: {title}")
    except Exception as e:
        logger.error(f"タイトルを取得中にエラーが発生しました: {e}")
        title = None

    return html_content, title

def extract_video_url(driver):
    try:
        sample_button = driver.find_element(By.XPATH, '//a[@onclick]')
        sample_button.click()
        time.sleep(5)

        logs = driver.get_log('performance')
        for log in logs:
            message = json.loads(log['message'])['message']
            if 'Network.responseReceived' in message['method']:
                response = message.get('params', {}).get('response', {})
                url = response.get('url', '')
                if url.endswith('.mp4'):
                    logger.info(f"動画のダウンロードURL: {url}")
                    return url

        logger.error("動画のURLが見つかりませんでした。")
        st.error("動画のURLが見つかりませんでした。")
        return None
    except Exception as e:
        logger.error(f"動画のURL抽出中にエラーが発生しました: {e}")
        st.error("動画のURL抽出中にエラーが発生しました。")
        return None

def download_video(video_url, output_path):
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for data in response.iter_content(1024):
                f.write(data)
        logger.info(f"動画を {output_path} として保存しました。")
        return output_path
    except requests.RequestException as e:
        logger.error(f"動画のダウンロード中にエラーが発生しました: {e}")
        st.error("動画のダウンロード中にエラーが発生しました。")
        return None

def search_videos(api_id, affiliate_id, keyword, sort, floor_code, hits=10, offset=1):
    params = {
        "api_id": api_id,
        "affiliate_id": affiliate_id,
        "site": "FANZA",
        "service": "digital",
        "floor": floor_code,
        "hits": hits,
        "offset": offset,
        "sort": sort,
        "keyword": keyword,
        "output": "json"
    }
    
    api_url = "https://api.dmm.com/affiliate/v3/ItemList"
    logger.info(f"API URL: {api_url} with params: {params}")
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"API Response: {data}")
        
        if 'result' not in data:
            logger.error("APIレスポンスに'result'キーがありません")
            st.error("APIレスポンスの形式が不正です")
            return []
        
        if 'items' not in data['result'] or not data['result']['items']:
            logger.info("検索結果が0件です")
            st.info("該当する動画が見つかりませんでした。検索条件を変更してお試しください。")
            return []
        
        return data['result']['items']
    
    except requests.RequestException as e:
        logger.error(f"APIリクエストエラー: {e}")
        st.error(f"APIリクエストエラー: {e}")
        return []
    except ValueError as e:
        logger.error(f"JSONデコードエラー: {e}")
        st.error("APIレスポンスの解析に失敗しました")
        return []

def calculate_skin_exposure_score(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([20, 150, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_area = cv2.countNonZero(skin_mask)
    total_area = frame.shape[0] * frame.shape[1]
    return skin_area / total_area

def extract_highlight_segments(video_path, segment_duration=15, top_n=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    step = int(segment_duration * fps)
    segments = []

    for start_frame in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        score_sum = 0
        frames_counted = 0

        for _ in range(step):
            ret, frame = cap.read()
            if not ret:
                break
            score_sum += calculate_skin_exposure_score(frame)
            frames_counted += 1

        if frames_counted > 0:
            average_score = score_sum / frames_counted
            segments.append(
                (start_frame / fps, min(segment_duration, total_duration - start_frame / fps), average_score))

    cap.release()
    segments.sort(key=lambda x: x[2], reverse=True)
    return segments[:top_n]

def save_segment_clips(video_path, segments, output_dir, save_segments=False):
    ensure_directory_exists(output_dir)
    clips = []
    temp_dir = os.path.join(output_dir, "temp_segments")
    ensure_directory_exists(temp_dir)
    used_segments = set()

    with VideoFileClip(video_path) as video:
        for i, (start, duration, _) in enumerate(segments):
            if start in used_segments:
                continue
            end = min(start + duration, video.duration)
            clip = video.subclip(start, end)
            try:
                clip.audio = clip.audio.set_fps(44100)
            except AttributeError:
                clip = clip.without_audio()
            clip_path = os.path.join(output_dir if save_segments else temp_dir, f"segment_{i + 1}.mp4")
            clip.write_videofile(clip_path, codec="libx264", audio_codec='aac')
            clips.append(clip_path)
            used_segments.add(start)

    return clips, temp_dir

def create_highlight_video(segment_paths, output_path, max_duration=140):
    clips = [VideoFileClip(segment) for segment in segment_paths]
    final_clip = concatenate_videoclips(clips)

    if final_clip.duration > max_duration:
        final_clip = final_clip.subclip(0, max_duration)

    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    final_clip.close()

    for clip in clips:
        clip.close()

def detect_highlight_scenes(video_path, output_dir, segment_duration=15, top_n=5, max_duration=140, save_segments=False):
    highlight_segments = extract_highlight_segments(video_path, segment_duration, top_n)
    segment_paths, _ = save_segment_clips(video_path, highlight_segments, output_dir, save_segments)
    highlight_output_path = os.path.join(output_dir, f"{Path(video_path).stem}_highlight.mp4")
    create_highlight_video(segment_paths, highlight_output_path, max_duration)
    logger.info(f"ハイライト動画を {highlight_output_path} として保存しました。")

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def open_explorer_and_bring_to_front(path):
    if platform.system() == "Darwin":  # macOS
        subprocess.Popen(['open', os.path.abspath(path)])
    elif platform.system() == "Windows":  # Windows
        subprocess.Popen(['explorer', os.path.abspath(path)])
    else:
        logger.error(f"この機能は {platform.system()} ではサポートされていません。")

def download_and_process_video(sample_url, affiliate_url, title, output_base_dir):
    driver = setup_driver()
    html_content, title = fetch_video_page(driver, sample_url)
    video_url = extract_video_url(driver)
    if video_url:
        output_dir = output_base_dir / title
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded_video_path = str(output_dir / f"{title}.mp4")
        downloaded_video = download_video(video_url, downloaded_video_path)
        if downloaded_video and os.path.getsize(downloaded_video) > 100 * 1024:
            logger.info(f"動画のダウンロードに成功しました: {downloaded_video}")
            fixed_video = fix_video_file(downloaded_video)
            if fixed_video:
                detect_highlight_scenes(fixed_video, output_dir)
            else:
                logger.error("動画ファイルの修正に失敗しました。")
                st.error("動画ファイルの修正に失敗しました。")
            affiliate_url_path = output_dir / "affiliate_url.txt"
            with open(affiliate_url_path, 'w') as f:
                f.write(affiliate_url)
            logger.info(f"アフィリエイトURLを {affiliate_url_path} に保存しました。")
            open_explorer_and_bring_to_front(output_dir)
        else:
            logger.error("動画のダウンロードに失敗しました。ファイルが無効です。")
            st.error("動画のダウンロードに失敗しました。ファイルが無効です。")
    else:
        logger.error("動画のURLが見つかりませんでした。")
        st.error("動画のURLが見つかりませんでした。")
    driver.quit()

def main():
    st.set_page_config(layout="wide")  # Wideモードに設定
    st.title("Sample Video Downloader")
    
    with st.sidebar:
        st.markdown('Powered by <a href="https://affiliate.dmm.com/api/">FANZA Webサービス</a>', unsafe_allow_html=True)
        api_id = st.text_input("API ID", value=config['api_id'], type="password")
        affiliate_id = st.text_input("Affiliate ID", value=config['affiliate_id'], type="password")
        
        if st.button("認証情報を保存"):
            save_config(api_id, affiliate_id)  # API IDとアフィリエイトIDを保存
        else:
            pass
        
        keyword = st.text_input("キーワード検索")
        
        sort_options = {
            "人気": "rank",
            "価格が高い順": "price",
            "価格が安い順": "-price",
            "新着": "date",
            "評価": "review",
            "マッチング順": "match"
        }
        sort = st.selectbox("ソート順", list(sort_options.keys()))
        selected_sort = sort_options[sort]

        floor_options = {
            "ビデオ": "videoa",
            "素人": "videoc",
            "成人映画": "nikkatsu",
            "アニメ動画": "anime"
        }
        floor = st.selectbox("フロアを選択してください", list(floor_options.keys()))
        selected_floor = floor_options[floor]
        
        hits = st.number_input("検索数", min_value=1, max_value=100, value=10)
        offset = st.number_input("オフセット", min_value=1, value=1)
        
    output_base_dir = Path.home() / "Desktop" / "video"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if st.button("商品データを取得する"):
        videos = search_videos(api_id, affiliate_id, keyword, selected_sort, selected_floor, hits, offset)
        
        if not videos:
            st.info("該当する動画が見つかりませんでした。検索条件を変更してお試しください。")
        else:
            st.success(f"{len(videos)}件の動画が見つかりました。")
            for i, video in enumerate(videos):
                with st.expander(f"{i+1}. {video['title']}"):
                    st.image(video['imageURL']['large'], use_column_width=True)
                    st.write(f"対象サービス: {video['service_name']}")
                    st.write(f"販売価格: {video['prices']['price']}円")
                    if 'URL' in video:
                        sample_url = video['URL']
                        affiliate_url = video['affiliateURL']
                        st.button("サンプル動画をダウンロードする", key=f"full_dl_button_{i}", on_click=download_and_process_video, args=(sample_url, affiliate_url, video['title'], output_base_dir))
                        st.button("ハイライト動画をダウンロードする", key=f"highlight_dl_button_{i}", on_click=download_and_process_video, args=(sample_url, affiliate_url, video['title'], output_base_dir))


if __name__ == '__main__':
    main()
