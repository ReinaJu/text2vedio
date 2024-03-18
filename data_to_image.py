# -*- coding: utf-8 -*-
"""
作者：张贵发
日期：2023年06月12日
描述：根据生成的prompt提示词来生成对应的图片
"""
import base64
import json
import os
import io
from PIL import Image
import pandas as pd
import requests
import shutil

from moviepy.video.compositing.transitions import crossfadein, crossfadeout
from moviepy.video.fx import fadein, fadeout

import utils
import config
from tqdm import tqdm
from moviepy.editor import *
from moviepy.video.tools.subtitles import *

def bf_draw_picture(idx):
    with open(config.image_desc_path,"r",encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(lines)
    if not os.path.exists(config.data_img_path):
        os.makedirs(config.data_img_path)
    sub_data_img_path = os.path.join(config.data_img_path,str(idx))
    if not os.path.exists(sub_data_img_path):
        os.makedirs(sub_data_img_path)
    else:
        shutil.rmtree(sub_data_img_path)
        os.makedirs(sub_data_img_path)
    for i,line in tqdm(enumerate(lines)):
        prompt = config.prompt+line
        novel_dict = {
            "enable_hr": "false",
            "denoising_strength": 0,
            "firstphase_width": 0,
            "firstphase_height": 0,
            "hr_scale": 2,
            "hr_upscaler": "string",
            "hr_second_pass_steps": 0,
            "hr_resize_x": 0,
            "hr_resize_y": 0,
            "prompt": prompt,
            "styles": [
                "string"
            ],
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "sampler_name": "DPM++ SDE Karras",
            "batch_size": 1,
            "n_iter": 1,
            "steps": 50,
            "cfg_scale": 7,
            "width": 1024,
            "height": 768,
            "restore_faces": "false",
            "tiling": "false",
            "do_not_save_samples": "false",
            "do_not_save_grid": "false",
            "negative_prompt": config.negative_prompt,
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {},
            "override_settings_restore_afterwards": "true",
            "script_args": [],
            "sampler_index": "DPM++ SDE Karras",
            "script_name": "",
            "send_images": "true",
            "save_images": "true",
            "alwayson_scripts": {}
        }

        sd_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"
        html = requests.post(sd_url, data=json.dumps(novel_dict))
        img_response = json.loads(html.text)
        image_bytes = base64.b64decode(img_response['images'][0])
        image = Image.open(io.BytesIO(image_bytes))
        image_path = os.path.join(sub_data_img_path,str(i)+".png")
        image.save(image_path)
    return sub_data_img_path

def fl_right(gf, t):
    # 获取原始图像帧
    frame = gf(t)

    # 进行滚动效果，将图像向右滚动50像素
    scroll_x = int(t * 1)  # 根据时间t计算滚动的像素数
    new_frame = np.zeros_like(frame)

    # 将原始帧的内容向右滚动50像素，并将结果赋值给新帧
    new_frame[:, scroll_x:] = frame[:, :frame.shape[1] - scroll_x]

    return new_frame

def bf_image_video(idx,image_dir_path,audio_dir_path):
    audio_files = os.listdir(audio_dir_path)
    # image_files = os.listdir(image_dir_path)
    clips = []
    for i in range(len(audio_files)):
        image_path = os.path.join(image_dir_path, str(i)+".png")
        audio_path = os.path.join(audio_dir_path, str(i)+".mp3")
        audio_clip = AudioFileClip(audio_path)
        img_clip = ImageSequenceClip([image_path], audio_clip.duration)
        img_clip = img_clip.set_position(('center', 'center')).fl(fl_right,apply_to=['mask']).set_duration(audio_clip.duration)
        clip = img_clip.set_audio(audio_clip)
        clips.append(clip)
    final_clip = concatenate_videoclips(clips)

    # 生成视频文件地址
    sub_data_video_path = utils.make_path(idx,config.data_video_path)
    video_file_path = os.path.join(sub_data_video_path, f"{str(idx)}.mp4")
    final_clip.write_videofile(video_file_path, fps=24,audio_codec="aac")
    return video_file_path,sub_data_video_path

def one2duo_image_video(idx,image_dir_path,audio_dir_path):
    audio_path = os.path.join(audio_dir_path, "1.wav")
    audio_clip = AudioFileClip(audio_path)
    audio_files = os.listdir(audio_dir_path)
    image_files = os.listdir(image_dir_path)
    # num_images = len(image_files)
    num_images = count_files_in_folder(image_dir_path)
    print(num_images)
    image_duration = round(audio_clip.duration / num_images, 1)
    print(image_duration)
    clips = []
    for i in range(num_images):
        image_path = convert_jpg_to_png(i, image_dir_path)
        # image_path = os.path.join(image_dir_path, str(i) + ".jpg")
        start_time = i * image_duration
        end_time = (i + 1) * image_duration
        if end_time > audio_clip.duration:
            # 如果最后一张图片的结束时间超出音频剪辑的长度，则将其设置为音频剪辑的结束时间
            end_time = audio_clip.duration
        # img_clip = ImageSequenceClip([image_path], end_time - start_time)
        # img_clip = img_clip.set_start(start_time).set_audio(audio_clip.subclip(start_time, end_time))
        # img_clip = img_clip.set_position(('center', 'center')).set_audio(audio_clip)
        if int((end_time - start_time) / image_duration) == 0:
            image_list = [image_path]
        else:
            image_list = [image_path] * int((end_time - start_time) / image_duration)
        # # 将 JPG 图像转换为 RGB 模式
        # image_list_rgb = [img.convert("RGB") for img in image_list]
        img_clip = ImageSequenceClip(image_list, durations=[image_duration] * len(image_list), ismask=True)
        audio_subclip = audio_clip.subclip(start_time, end_time)
        clip = CompositeVideoClip([img_clip.set_audio(audio_subclip)]).set_start(start_time)
        clip = clip.fx(vfx.fadein, duration=0.3).fx(vfx.fadeout, duration=0.3)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips)
    final_clip = final_clip.fx(vfx.speedx, factor=0.9)
    # final_clip = final_clip.fx(vfx.fadein, duration=1)  # 添加淡入效果，持续1秒
    # final_clip = final_clip.fx(vfx.fadeout, duration=1)  # 添加淡出效果，持续1秒

    # 生成视频文件地址
    sub_data_video_path = utils.make_path(idx,config.data_video_path)
    video_file_path = os.path.join(sub_data_video_path, f"{str(idx)}.mp4")
    final_clip.write_videofile(video_file_path, fps=24,audio_codec="aac")
    return video_file_path,sub_data_video_path

def convert_jpg_to_png(i, image_path):
    jpg_path = os.path.join(image_path, str(i) + ".jpg")
    target_folder = os.path.join(image_path, "png")
    print(target_folder)
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)
    png_path = os.path.join(target_folder, str(i) + ".png")
    try:
        # 打开 JPG 图片
        with Image.open(jpg_path) as im:
            # 将图像转换为 RGBA 格式
            converted_img = im.convert("RGBA")
            converted_img.save(png_path, "PNG", quality=2, optimize=True)
        print("转换成功！")
    except IOError:
        print("转换失败！")
    return png_path

def count_files_in_folder(folder_path):
    file_count = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_count += 1
    return file_count

if __name__ == '__main__':
    # draw_picture("data/data_prompt/侦探悬疑类/story_1.csv")
    # bf_draw_picture(1)
    # one2duo_image_video(0, r"C:\1-study\AI\text_to_vedio-master\text_to_vedio-master\data\data_img\test",
    #                r"C:\1-study\AI\text_to_vedio-master\text_to_vedio-master\data\data_voice\0")

    # convert_jpg_to_png(0, r"C:\1-study\AI\text_to_vedio-master\text_to_vedio-master\data\data_img\test")
    # 创建 JPG 图片对应的 ImageClip 对象
    image_clip_jpg = ImageClip("0.jpg")

    # 创建 PNG 图片对应的 ImageClip 对象
    image_clip_png = ImageClip("student.png")

    # 创建 CompositeVideoClip 对象并合成视频
    clip = CompositeVideoClip([image_clip_jpg, image_clip_png])
    clip.fps = 24
    # 设置视频剪辑的持续时间
    video_clip = clip.set_duration(10)  # 这里的参数 10 表示持续时间为 10 秒

    # 导出视频
    video_clip.write_videofile("output.mp4")


