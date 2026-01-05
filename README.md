# Campus QA Bot API

FastAPI untuk Campus QA Bot yang menggunakan model JointBERT untuk intent classification dan slot filling.

## Deployment Folder

Folder `deployment/` berisi output dari training model yang dilakukan di Google Colab:
https://colab.research.google.com/drive/12v6kJqd92MMERPXz9pEVFzZ9f22AO1nc

Folder ini mencakup:
- `model_quantized.pt` - Model yang sudah dikuantisasi (INT8) untuk inferensi lebih cepat
- `tokenizer/` - Tokenizer dari IndoBERT
- `label_config.json` - Konfigurasi label untuk intents dan slots

## Model

- **Base Model**: IndoBERT-base-p1 (indobenchmark/indobert-base-p1)
- **Architecture**: JointBERT untuk intent classification dan slot filling
- **Intents**: 4 intents (get_info_dosen, get_jadwal, get_ruangan, get_tugas_untuk_mata_kuliah)
- **Slots**: MATKUL, NAMA_DOSEN, WAKTU

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan API

```bash
uvicorn app:app --reload
```

API akan berjalan di http://localhost:8000

## API Endpoints

### `POST /predict`

Melakukan prediksi intent dan slot filling pada teks input.

**Request Body:**
```json
{
  "text": "jadwal kuliah IF4030 hari selasa"
}
```

**Response:**
```json
{
  "text": "jadwal kuliah IF4030 hari selasa",
  "intent": "get_jadwal",
  "confidence": 0.9876,
  "entities": [
    {
      "entity": "MATKUL",
      "text": "IF4030"
    },
    {
      "entity": "WAKTU",
      "text": "hari selasa"
    }
  ]
}
```

### `GET /`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "JointBERT Campus Bot"
}
```

## Struktur Project

```
api/
├── app.py                   # FastAPI application dan model definition
├── requirements.txt         # Python dependencies
├── deployment/             # Artifacts model dari Colab
│   ├── model_quantized.pt # Model weights (terkuantisasi)
│   ├── tokenizer/         # Tokenizer files
│   ├── label_config.json  # Label configuration
│   └── classification_heads.pt
└── README.md
```
