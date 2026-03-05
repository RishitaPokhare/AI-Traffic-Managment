import cv2
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────
MODEL_PATH = "best.pt"
VIDEO_PATH = "videos/high.mp4"
OUTPUT     = "videos/output_high.mp4"

# ── Load ───────────────────────────────────────────────
model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps   = cap.get(cv2.CAP_PROP_FPS)
out   = cv2.VideoWriter(OUTPUT,
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# ── Counting line ──────────────────────────────────────
LINE_Y_RATIO = 0.85
LINE_Y = int(h * LINE_Y_RATIO)

# ── Variables ──────────────────────────────────────────
counted_ids     = set()
vehicle_count   = 0
last_cy_by_id   = {}
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}
COLORS = {
    'car':        (0,   200, 255),
    'bus':        (0,   255, 100),
    'truck':      (255, 100,   0),
    'motorcycle': (200,   0, 255),
}

MODEL_NAMES = model.names if hasattr(model, "names") else {}
VEHICLE_KEYWORDS = (
    "car", "bus", "truck", "motorcycle", "motorbike",
    "bike", "bicycle", "van", "vehicle", "auto", "rickshaw"
)

def get_model_label(cls_id):
    if isinstance(MODEL_NAMES, dict):
        return str(MODEL_NAMES.get(int(cls_id), "vehicle")).lower()
    if isinstance(MODEL_NAMES, list) and 0 <= int(cls_id) < len(MODEL_NAMES):
        return str(MODEL_NAMES[int(cls_id)]).lower()
    return "vehicle"

detected_vehicle_classes = []
if isinstance(MODEL_NAMES, dict):
    detected_vehicle_classes = [
        idx for idx, name in MODEL_NAMES.items()
        if any(k in str(name).lower() for k in VEHICLE_KEYWORDS)
    ]
elif isinstance(MODEL_NAMES, list):
    detected_vehicle_classes = [
        idx for idx, name in enumerate(MODEL_NAMES)
        if any(k in str(name).lower() for k in VEHICLE_KEYWORDS)
    ]

if detected_vehicle_classes:
    TRACK_CLASSES = sorted(set(int(i) for i in detected_vehicle_classes))
else:
    TRACK_CLASSES = VEHICLE_CLASSES

# ── Signal logic ───────────────────────────────────────
def get_signal(count):
    if count <= 5:
        return "LOW",      (0, 255, 100), 15
    elif count <= 15:
        return "MEDIUM",   (0, 220, 255), 30
    elif count <= 25:
        return "HIGH",     (0, 140, 255), 45
    else:
        return "CRITICAL", (0,   0, 255), 60

# ── Draw info box ──────────────────────────────────────
def draw_hud(frame, total_counted, in_frame):
    density, color, green_time = get_signal(in_frame)
    cv2.rectangle(frame, (0,0), (420,190), (0,0,0), -1)
    cv2.rectangle(frame, (0,0), (420,190), color, 2)
    cv2.putText(frame, f'TOTAL COUNTED    : {total_counted}',
        (10,38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255,255,255), 2)
    cv2.putText(frame, f'IN FRAME NOW      : {in_frame}',
        (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255,255,255), 2)
    cv2.putText(frame, f'DENSITY           : {density}',
        (10,112), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        color, 2)
    cv2.putText(frame, f'GREEN SIGNAL      : {green_time} SEC',
        (10,149), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0,255,100), 2)
    bar = min(int((in_frame/40)*400), 400)
    cv2.rectangle(frame,(10,162),(410,180),(50,50,50),-1)
    cv2.rectangle(frame,(10,162),(10+bar,180), color,-1)

# ── Draw traffic light ─────────────────────────────────
def draw_light(frame, in_frame):
    _, color, green_time = get_signal(in_frame)
    lx, ly = w-90, 10
    cv2.rectangle(frame,(lx-10,ly),(lx+60,ly+170),
                  (30,30,30),-1)
    cv2.rectangle(frame,(lx-10,ly),(lx+60,ly+170),
                  (150,150,150),2)
    r = (0,0,255)   if in_frame > 25  else (40,0,40)
    y = (0,200,255) if 15 < in_frame <= 25 else (40,40,0)
    g = (0,255,100) if in_frame <= 15 else (0,40,0)
    cv2.circle(frame,(lx+20,ly+28), 22,r,-1)
    cv2.circle(frame,(lx+20,ly+85), 22,y,-1)
    cv2.circle(frame,(lx+20,ly+142),22,g,-1)
    cv2.putText(frame,f'{green_time}s',
        (lx+5,ly+168),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,(255,255,255),1)

print("Running... Press Q to quit")
print(f"Tracking classes: {TRACK_CLASSES}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ── Track ──────────────────────────────────────────
    results = model.track(
        frame,
        persist = True,
        conf    = 0.15,
        iou     = 0.45,
        classes = TRACK_CLASSES,
        tracker = "bytetrack.yaml",
        verbose = False
    )

    # ── Counting line ──────────────────────────────────
    cv2.line(frame,(0,LINE_Y),(w,LINE_Y),(0,255,255),3)
    cv2.putText(frame,'COUNTING LINE',
        (10,LINE_Y-12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,(0,255,255),2)
    frame_vehicles = 0

    if results[0].boxes is not None:

        if results[0].boxes.id is not None:
            # ── Tracking working ───────────────────────
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs   = results[0].boxes.conf.cpu().numpy()

            for box,track_id,cls,conf in zip(
                    boxes,ids,classes,confs):
                x1,y1,x2,y2 = map(int,box)
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                label = CLASS_NAMES.get(cls, get_model_label(cls))
                color = COLORS.get(label,(128,128,128))
                frame_vehicles += 1

                prev_cy = last_cy_by_id.get(track_id)
                last_cy_by_id[track_id] = cy
                crossed_line = (
                    prev_cy is not None and (
                        (prev_cy < LINE_Y <= cy) or
                        (prev_cy > LINE_Y >= cy)
                    )
                )

                if (crossed_line and
                        track_id not in counted_ids):
                    counted_ids.add(track_id)
                    vehicle_count += 1
                    print(f"Counted line-cross! "
                          f"ID:{track_id} "
                          f"{label} "
                          f"Total:{vehicle_count}")

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,
                    f'{label} ID:{track_id}',
                    (x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,color,1)
                cv2.circle(frame,(cx,cy),5,color,-1)

        else:
            # ── Fallback if no IDs ─────────────────────
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs   = results[0].boxes.conf.cpu().numpy()

            for box,cls,conf in zip(boxes,classes,confs):
                x1,y1,x2,y2 = map(int,box)
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                label = CLASS_NAMES.get(cls, get_model_label(cls))
                color = COLORS.get(label,(128,128,128))
                frame_vehicles += 1

                key = f"{cx//40}_{cy//40}"
                if cy > LINE_Y and key not in counted_ids:
                    counted_ids.add(key)
                    vehicle_count += 1
                    print(f"Counted fallback! {label} "
                          f"Total:{vehicle_count}")

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,
                    f'{label} {conf:.2f}',
                    (x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,color,1)
                cv2.circle(frame,(cx,cy),5,color,-1)

    # ── Draw HUD ───────────────────────────────────────
    draw_hud(frame, vehicle_count, frame_vehicles)
    draw_light(frame, frame_vehicles)

    cv2.imshow('Traffic Counter', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nDONE!")
print(f"Total vehicles counted : {vehicle_count}")
print(f"Output saved           : {OUTPUT}")
