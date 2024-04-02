import os
from PIL import Image
from ultralytics import YOLO

model = YOLO('井盖测试集/best.pt')

source = '井盖测试集/测试集图片'

results = model(source, stream=True)

result_dir = 'test_results'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    save_path = os.path.join(result_dir, f'results{i}.jpg')
    im_rgb.save(save_path)
