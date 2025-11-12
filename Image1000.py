import random
from PIL import Image
import os

# --- 🎈 여기를 필요에 맞게 수정하세요 ---

# 1. 해충 정보: 파일 이름, 클래스 ID, 실제 크기(mm) 범위 설정
# (클래스 ID는 YOLO 라벨링에 사용됨)
pest_info = {
    'Aphid.png':     {'id': 0, 'size': (1.0, 2.0)}, # 클래스 ID 0
    'mite.png':      {'id': 1, 'size': (1.0, 1.0)}, # 클래스 ID 1
    'Thrips.png':    {'id': 2, 'size': (1.0, 2.0)}, # 클래스 ID 2
    'whitefly1.png': {'id': 3, 'size': (1.5, 2.0)}, # 클래스 ID 3
    'whitefly2.png': {'id': 3, 'size': (1.5, 2.0)}  # 클래스 ID 3
}

# 2. 생성할 총 이미지 개수 (훈련 800장, 검증 200장)
num_images_to_generate = 1000
# 3. 이 중에서 검증(validation)용으로 쓸 개수
num_val_images = 200 

# 4. 파일 이름 (앞부분)
output_prefix = 'trap_640_v1' # 버전 관리용 이름

# 5. 해충 개수 범위 (640x640 이미지에 맞게 조절)
min_pests = 10
max_pests = 50

# 6. 저장할 폴더 이름
base_dir = 'dataset_640'

# --- 수정 끝 ---

# --- 폴더 생성 로직 ---
# dataset_640/images/train, dataset_640/labels/train 등 폴더를 자동으로 만듭니다.
img_train_dir = os.path.join(base_dir, 'images', 'train')
lbl_train_dir = os.path.join(base_dir, 'labels', 'train')
img_val_dir = os.path.join(base_dir, 'images', 'val')
lbl_val_dir = os.path.join(base_dir, 'labels', 'val')

os.makedirs(img_train_dir, exist_ok=True)
os.makedirs(lbl_train_dir, exist_ok=True)
os.makedirs(img_val_dir, exist_ok=True)
os.makedirs(lbl_val_dir, exist_ok=True)
print(f"'{base_dir}' 폴더 구조 생성 완료.")

# --- (1) 640x640 이미지 생성 로직 ---
bg_width_px = 640
bg_height_px = 640
YELLOW_HEX = "#FFFF00"
REAL_TRAP_WIDTH_MM = 64.0 # 6.4cm로 가정
pixels_per_mm = bg_width_px / REAL_TRAP_WIDTH_MM

print(f"생성할 배경 이미지 크기: {bg_width_px}x{bg_height_px} 픽셀")
print(f"1mm당 픽셀 비율: 약 {pixels_per_mm:.2f} 픽셀/mm")

# --- 해충 이미지 미리 불러오기 ---
try:
    pest_images = {name: Image.open(name).convert("RGBA") for name in pest_info.keys()}
except FileNotFoundError as e:
    print(f"🚨 오류: 해충 파일을 찾을 수 없습니다! -> {e}")
    print("스크립트와 해충 .png 파일이 같은 폴더에 있는지 확인하세요.")
    exit()

pest_names_list = list(pest_info.keys())

# --- 메인 루프: 지정된 개수만큼 이미지 생성 ---
for i in range(1, num_images_to_generate + 1):
    
    # 훈련용/검증용 폴더 자동 분배
    if i <= (num_images_to_generate - num_val_images):
        current_img_dir = img_train_dir
        current_lbl_dir = lbl_train_dir
        mode = "훈련(train)"
    else:
        current_img_dir = img_val_dir
        current_lbl_dir = lbl_val_dir
        mode = "검증(val)"

    num_pests_to_add = random.randint(min_pests, max_pests)
    
    if i % 50 == 0 or i == 1: # 50장마다 또는 첫 장에 로그 표시
        print(f"--- 🖼️  {mode} 이미지 {i}/{num_images_to_generate} 생성 중 (해충 {num_pests_to_add}마리) ---")

    # 1. 640x640 노란색 배경 생성
    background = Image.new("RGBA", (bg_width_px, bg_height_px), YELLOW_HEX)
    labels_for_this_image = []

    # 2. 해충을 무작위로 배치
    for _ in range(num_pests_to_add):
        pest_name = random.choice(pest_names_list)
        pest_image = pest_images[pest_name]
        
        class_id = pest_info[pest_name]['id'] # 라벨링을 위한 ID
        min_mm, max_mm = pest_info[pest_name]['size']

        target_size_mm = random.uniform(min_mm, max_mm)
        target_pixel_width = int(target_size_mm * pixels_per_mm)
        
        original_width, original_height = pest_image.size
        aspect_ratio = original_height / original_width
        target_pixel_height = int(target_pixel_width * aspect_ratio)
        
        if target_pixel_width <= 0 or target_pixel_height <= 0:
            continue
            
        # ✨✨✨ 화질 개선 옵션(LANCZOS)이 제거된 부분 ✨✨✨
        resized_pest = pest_image.resize(
            (target_pixel_width, target_pixel_height)
            # resample 옵션 없이 라이브러리 기본값을 사용합니다.
        )
        
        angle = random.randint(0, 360)
        rotated_pest = resized_pest.rotate(angle, expand=True)

        max_x = bg_width_px - rotated_pest.width
        max_y = bg_height_px - rotated_pest.height
        
        if max_x < 0 or max_y < 0:
            continue

        rand_x = random.randint(0, max_x) # 붙일 x좌표 (좌측 상단)
        rand_y = random.randint(0, max_y) # 붙일 y좌표 (좌측 상단)

        # 3. 배경에 해충 이미지 붙이기
        background.paste(rotated_pest, (rand_x, rand_y), rotated_pest)

        # --- ⭐️ (2) YOLO 라벨링 코드 ⭐️ ---
        # 방금 붙인 해충의 좌표와 크기로 YOLO 라벨을 계산
        
        final_w = rotated_pest.width
        final_h = rotated_pest.height
        
        # 바운딩 박스의 중심점 좌표 (픽셀)
        center_x_px = rand_x + (final_w / 2)
        center_y_px = rand_y + (final_h / 2)

        # 0~1 사이 값으로 정규화(Normalize)
        x_center_norm = center_x_px / bg_width_px
        y_center_norm = center_y_px / bg_height_px
        width_norm = final_w / bg_width_px
        height_norm = final_h / bg_height_px

        # YOLO 라벨 형식: "class_id x_center y_center width height"
        label_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
        labels_for_this_image.append(label_line)
        # --- 라벨링 코드 끝 ---


    # 4. 최종 결과물 저장 (이미지와 라벨 동시 저장)
    file_name = f"{output_prefix}_{i:04d}" # 0001, 0002... 형식
    output_image_path = os.path.join(current_img_dir, f"{file_name}.png")
    output_label_path = os.path.join(current_lbl_dir, f"{file_name}.txt")

    # 4-1. 이미지(.png) 저장
    background.save(output_image_path)
    
    # 4-2. 라벨(.txt) 파일 저장
    with open(output_label_path, 'w') as f:
        for line in labels_for_this_image:
            f.write(line + '\n') # 리스트에 담아뒀던 모든 라벨을 파일에 씀

print(f"🎉 모든 작업 완료! '{base_dir}' 폴더를 확인하세요.")