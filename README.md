# Copa FutBotMX — ArbyKing

**Categoría:** Amateur  
**Reto:** Visión por computadora con SAM 3  
**Equipo:** ArbyKing  
**Repositorio:** [github.com/RDGKing/arbyking-futbotmx](https://github.com/RDGKing/arbyking-futbotmx)  
**Notebook:** [notebooks/arby_futbot_mx.ipynb](notebooks/arby_futbot_mx.ipynb)

Pipeline de visión por computadora para analizar partidos de fútbol robótico: detección de robots y pelota, tracking multi-objeto, segmentación con SAM 3, video comparativo y mapa de calor de actividad.

---

## Enfoque

```
Video del partido
      ↓
YOLO custom (Robot + Pelota)     ← fine-tune con dataset etiquetado en Roboflow
      ↓
ByteTrack (Supervision)          ← IDs estables frame a frame
      ↓
SAM 3 (Ultralytics)              ← máscaras dentro de cada caja de YOLO
      ↓
Video demo lado a lado + heatmap de actividad de robots
```

YOLO base (COCO) no detecta robots FutBotMX. Entrené un `yolov8n` con frames propios del torneo. ByteTrack mantiene la identidad de cada robot entre frames. SAM 3 refina la silueta usando las cajas de YOLO como prompt visual, sin reentrenar el modelo.

---

## Requisitos

| Componente | Recomendado | Mínimo |
|------------|-------------|--------|
| Python | 3.11 | 3.10+ |
| GPU | NVIDIA con CUDA | CPU (pruebas cortas) |
| RAM | 16 GB | 8 GB |

Dependencias principales: `ultralytics`, `supervision`, `opencv-python`, `matplotlib`, `numpy`. Detalle completo en [requirements.txt](requirements.txt).

---

## Instalación

### Google Colab

```bash
!pip install ultralytics supervision opencv-python-headless matplotlib numpy
```

Clonar o subir el repo, descargar `sam3.pt` en `assets/models/` y ejecutar el notebook de arriba hacia abajo.

### Local

```bash
git clone https://github.com/RDGKing/arbyking-futbotmx.git
cd arbyking-futbotmx
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Con conda: `conda env create -f environment.yml && conda activate futbot`

### Modelos

| Archivo | En Git | Notas |
|---------|--------|-------|
| `assets/models/yolov8n.pt` | Sí | Base YOLO |
| `assets/models/yolo_futbot/weights/best.pt` | Sí | Detector entrenado |
| `assets/models/sam3.pt` | No | [Descargar SAM 3](https://huggingface.co/facebook/sam3) — excede límite de GitHub |

### Videos fuente

La mayoría de los partidos están en `assets/videos/`. Dos archivos superan el límite de **100 MB** de GitHub y no están en este repositorio:

| Archivo | Tamaño | Disponible en |
|---------|--------|---------------|
| `IMG_9822.MOV` | 171 MB | [Google Drive — arbyking-futbotmx-videos](https://drive.google.com/drive/folders/1LyBSD68Q2XQPClMj0USCZeBASstfYTvt) |
| `IMG_9863.MOV` | 206 MB | [Google Drive — arbyking-futbotmx-videos](https://drive.google.com/drive/folders/1LyBSD68Q2XQPClMj0USCZeBASstfYTvt) |

Descárgalos del Drive y colócalos en `assets/videos/` para reproducir el notebook con esos clips.

---

## Reproducción

1. Abrir [notebooks/arby_futbot_mx.ipynb](notebooks/arby_futbot_mx.ipynb).
2. Ejecutar celdas en orden:
   - **Variables y rutas** — video, modelos, salidas.
   - **1–2** — cargar video y probar YOLO base.
   - **3** — fine-tune YOLO (opcional si ya existe `best.pt`).
   - **4–6** — detección, ByteTrack y SAM en frames de prueba.
   - **7** — video demo y heatmap.
   - **8** — procesar otro video cambiando `VIDEO_IN`.
3. Videos fuente en `assets/videos/` (`IMG_9822.MOV` e `IMG_9863.MOV` en [Google Drive](https://drive.google.com/drive/folders/1LyBSD68Q2XQPClMj0USCZeBASstfYTvt)).

---

## Resultados

### Videos demo — original vs. análisis

| Demo | Enlace |
|------|--------|
| Demo principal | [demo_arbyking.mp4](assets/outputs/demo_arbyking.mp4) |
| Partido IMG_9812 | [demo_IMG_9812.mp4](assets/outputs/demo_IMG_9812.mp4) |
| Partido IMG_9834 | [demo_IMG_9834.mp4](assets/outputs/demo_IMG_9834.mp4) |

### Mapa de calor — IMG_9812

| Archivo | Enlace |
|---------|--------|
| Imagen PNG | [heatmap_robots_IMG_9812_heatmap.png](assets/images/heatmap_robots_IMG_9812_heatmap.png) |
| Video | [heatmap_robots_IMG_9812_heatmap.mp4](assets/videos/heatmap_robots_IMG_9812_heatmap.mp4) |

### Pipeline por etapas (frame de prueba)

| Etapa | Enlace |
|-------|--------|
| Frame original | [frame_00000.jpg](assets/outputs/frame_00000.jpg) |
| Detección YOLO custom | [deteccion_frame_00000.jpg](assets/outputs/deteccion_frame_00000.jpg) |
| Tracking ByteTrack | [tracking_frame_00059.jpg](assets/outputs/tracking_frame_00059.jpg) |
| Segmentación SAM 3 | [sam_frame_00000.jpg](assets/outputs/sam_frame_00000.jpg) |

---

## Reel de Instagram

[instagram.com/reel/DZwKpByM5vW](https://www.instagram.com/reel/DZwKpByM5vW/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA==)

---

## Estructura del repositorio

```
arbyking-futbotmx/
├── notebooks/
│   └── arby_futbot_mx.ipynb
├── assets/
│   ├── dataset/          # futbot.yaml, train/, valid/
│   ├── images/           # heatmaps
│   ├── models/           # yolov8n.pt, yolo_futbot/ (sam3.pt local)
│   ├── outputs/          # demos y frames
│   └── videos/           # partidos fuente
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md
```

---

## Licencia y créditos

### Código del proyecto

Licencia **MIT** — ver [LICENSE](LICENSE).

### SAM 3 (Meta)

Este proyecto utiliza **SAM 3** (Segment Anything Model 3) de Meta AI.

- Repositorio: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- Hugging Face: [facebook/sam3](https://huggingface.co/facebook/sam3)
- Licencia del modelo: [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)
- Paper: Carion et al. (2025). *SAM 3: Segment Anything with Concepts.* [arXiv:2511.16719](https://arxiv.org/abs/2511.16719)

### Otras dependencias

| Recurso | Licencia |
|---------|----------|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| [Roboflow Supervision](https://github.com/roboflow/supervision) | MIT |
| [OpenCV](https://opencv.org/) | Apache 2.0 |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | MIT |

### Datos

- Videos de partidos: repositorio oficial Copa FutBotMX / Federación Mexicana de Robótica.
- Dataset de entrenamiento: frames propios etiquetados en [Roboflow](https://roboflow.com) (clases `Robot` y `Pelota`).

---

**ArbyKing** · [Copa FutBotMX](https://secihti.mx/futbotmx/) · [GitHub](https://github.com/RDGKing/arbyking-futbotmx)
