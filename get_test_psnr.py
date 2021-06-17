import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from benchmark.pytorch_msssim import ssim_matlab
from dataset import *
import math

warnings.filterwarnings("ignore")

def transferAudio(sourceVideo, targetVideo):
    #把原视频的音频部分提取出来，与生成的插帧视频结合起来，生成的视频没有音频，temp directory暂时用于存放音频信息
    #如果ffmpeg无法将视频和音频合并在一起，尝试将音频转换为aac
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits覆盖
        if os.path.isdir("temp"):
            # remove temp directory删除
            shutil.rmtree("temp")
        # create new "temp" directory新建
        os.makedirs("temp")
        # extract audio from video 提取音频
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))#source提取音频，放到targetno音频中，生成最终的

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file合并音频与视频
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac如果ffmpeg无法将视频和音频合并在一起，请尝试将音频转换为aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# description - 在参数帮助文档之前显示的文本
# dest - 被添加到 parse_args() 所返回对象上的属性名。
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
args = parser.parse_args()
assert (not args.video is None or not args.img is None)
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 不进行梯度更新
torch.set_grad_enabled(False)
if torch.cuda.is_available():#加速网络，自动寻找最适合当前配置的高效算法
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    #设置默认tensor
    if(args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)#缓解GPU显存不足

try:
    from model.RIFE_HDv2 import Model
    model = Model()
    model.load_model(args.modelDir, -1)   #modelDir=train_log预训练模型存放,-1去掉不需要的key值"module."
    print("Loaded v2.x HD model.")
except:
    from model.RIFE_HD import Model
    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded v1.x HD model")
model.eval()#开始测试
model.device()#路径放到cuda or cpu

#从write_buffer队列（放lastframe是原始帧，'output'是经过inference处理后的插帧结果）里面获取帧写入vid_out输出视频容器中
#source video ：args.video, target video：vid_out_name

if not args.video is None:
    #打开视频
    videoCapture = cv2.VideoCapture(args.video)
    #获取fps
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    #获取帧数
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    #空间释放
    videoCapture.release()
    #不输入fps，按输入的exp插帧；输入fps不改变
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * (2 ** args.exp)
    else:
        fpsNotAssigned = False
    #逐帧加载视频，任意视频数据作为一个单独的ndarry加载入内存，全部，需要大内存
    videogen = skvideo.io.vreader(args.video)
    #下一帧，逐帧读
    lastframe = next(videogen)
    #视频编解码器，编码器mp4，是一个 32 位的无符号数值，用 4 个字母表示采用的编码器
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #将文件名和扩展名分开
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    #视频、无fps规定无慢动作要求、无skip，经过插值处理后，音频将被合并
    if args.png == False and fpsNotAssigned == True and not args.skip:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png, fps or skip flag!")
else:#图片，输入视频为空
    videogen = []
    for f in os.listdir(args.img):#返回文件夹列表
        if 'png' in f:#只处理 后缀png
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))#到倒数第四个停止，将‘.png’左边的字符转换成整数型进行排序
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]))[:, :, ::-1].copy()#倒序copy
    videogen = videogen[1:]#顺序输出
h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** args.exp), int(np.round(args.fps)), args.ext)
    #建立保存视频文件的容器，规定帧率，在一帧一帧写进去，帧数是生成的
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))#('要保存的文件名',fourcc编码器, 帧率, 像素(1920,1080),彩色True)

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        #逐帧读
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            #如果是图像，不变成视频，直接输出图像，写入vid_out文件夹，按顺序,一共7位数
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1#重新排序
        else:
            #视频获取插帧图像后，整合为视频
            #从write_buffer队列（放lastframe）里面获取帧写入vid_out输出视频中，-1倒着写
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
             if not user_args.img is None:
                  frame = cv2.imread(os.path.join(user_args.img, frame))[:, :, ::-1].copy()
             if user_args.montage:
                  frame = frame[:, left: left + w]
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):
    global model
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n//2)
    second_half = make_inference(middle, I1, n=n//2)
    if n%2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()#填充函数
    else:
        return F.pad(img, padding)

if args.montage:
    left = w // 4
    w = w // 2
tmp = max(32, int(32 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)#进度条
skip_frame = 1
if args.montage:
    lastframe = lastframe[:, left: left + w]
#初始化队列，缓冲区
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
#多线程
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))#把frame放到read_buffer里
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))#从write_buffer中读到读frame写到vid_out

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.#（2，0，1）通道，x，y,归一化
I1 = pad_image(I1)#填充

psnr_list = []
dataset_val = VimeoDataset('validation')  # 带有GT验证集，后20个
val_data = DataLoader(dataset_val, batch_size=12, pin_memory=True, num_workers=8)
# while True:
#     #逐帧读 出队，变成201张量归一化，分别(32, 32)下采样，计算两张图片相似度，相似就skip，差太多转场就融合，中间就插帧
#     frame = read_buffer.get()
for i, data in enumerate(val_data):
    data_gpu, flow_gt = data
    data_gpu = data_gpu.to(device, non_blocking=True) / 255.
    flow_gt = flow_gt.to(device, non_blocking=True)
    imgs = data_gpu[:, :6]#前2张
    gt = data_gpu[:, 6:9]#第三张

    I0 = imgs[:, :3]
    I1 = imgs[:, 3:]
    I0 = torch.from_numpy(np.transpose(I0, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I0 = pad_image(I0)
    I1 = torch.from_numpy(np.transpose(I1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    #上下采样，I0输出尺寸(32, 32)下采样，双线性插值，对齐方法
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    # ssim = ssim_matlab(I0_small, I1_small)

    # if ssim > 0.995:#如果输入skip，相邻两帧几乎一样就skip，认为其为static frame；没有输入print相似帧
    #     if skip_frame % 100 == 0:
    #         print("\n·Warning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
    #         #means that your video has changed the frame rate by adding static frames, it is common if you have processed 25FPS video to 30FPS.
    #     #     静态帧不处理，会少帧？？？？？？
    #     skip_frame += 1
    #     if args.skip:
    #         #每次更新进度条的长度
    #         #加一个输出
    #         print("%%%%%%%%%%%%%skip=true")
    #         pbar.update(1)
    #         continue
    #
    # if ssim < 0.5:
    #     print("ssim小于0。5")
    #     output = []
    #     step = 1 / (2 ** args.exp)
    #     alpha = 0
    #     for i in range((2 ** args.exp) - 1):
    #         alpha += step
    #         beta = 1-alpha
    #         # 转场，图像混合加权
    #         output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
    # else:
    print("ssim位于中间")
    pred = make_inference(I0, I1, 2**args.exp-1) if args.exp else []

    # 计算psnr
    for j in range(gt.shape[0]):
        psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
        psnr_list.append(psnr)
    print(np.array(psnr_list).mean())

    # if args.montage:#两张图拼接在一起
    #     write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    #     for mid in output:
    #         mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
    #         write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    # else:
    #     write_buffer.put(lastframe)
    #     for mid in output:
    #         mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
    #         write_buffer.put(mid[:h, :w])
    # pbar.update(1)#进度条
    # lastframe = frame
#
# if args.montage:
#     write_buffer.put(np.concatenate((lastframe, lastframe), 1))
# else:
#     write_buffer.put(lastframe)
# import time
# while(not write_buffer.empty()):
#     time.sleep(0.1)
# pbar.close()
# if not vid_out is None:
#     vid_out.release()
#
# # move audio to new video file if appropriate音频合并
# if args.png == False and fpsNotAssigned == True and not args.skip and not args.video is None:
#     try:
#         transferAudio(args.video, vid_out_name)
#     except:
#         print("Audio transfer failed. Interpolated video will have no audio")
#         targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
#         os.rename(targetNoAudio, vid_out_name)
