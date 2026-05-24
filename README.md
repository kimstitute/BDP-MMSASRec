# GMUSASRec Experiments

## 환경 설정

```bash
cd RecBole
pip install -e .
```

## 데이터셋 배치

아래 경로에 데이터셋을 배치한다.

```
/workspace/BDP/Datasets/
├── Appliances/
│   ├── Appliances.inter        # user_id:token  item_id:token  timestamp:float
│   └── Appliances.item         # item_id:token  img_feat:float_seq
├── Beauty_and_Personal_Care/
│   ├── Beauty_and_Personal_Care.inter
│   └── Beauty_and_Personal_Care.item
└── Clothing_Shoes_and_Jewelry/
    ├── Clothing_Shoes_and_Jewelry.inter
    └── Clothing_Shoes_and_Jewelry.item
```

`img_feat`는 2048차원 float 벡터(ResNet-50 fc 레이어 출력)다.

경로가 다를 경우 각 YAML의 `data_path`를 수정하거나 실행 시 `--data_path=<경로>`로 오버라이드한다.

## 실험 실행

```bash
cd RecBole
PYTHON=python
RUNNER=run_recbole.py
CONFIG=../configs/runs
```

### Appliances (MAX_ITEM_LIST_LENGTH=50, stopping_step=2)

```bash
# M1 SASRec
$PYTHON $RUNNER --model=SASRec --dataset=Appliances --config_files=$CONFIG/appliances_m1_fast.yaml

# M2 SASRecF
$PYTHON $RUNNER --model=SASRecF --dataset=Appliances --config_files=$CONFIG/appliances_m2_fast.yaml

# M3 GMUSASRec
$PYTHON $RUNNER --model=GMUSASRec --dataset=Appliances --config_files=$CONFIG/appliances_m3_fast.yaml
```

### Beauty_and_Personal_Care (MAX_ITEM_LIST_LENGTH=20, stopping_step=3)

```bash
# M1 SASRec
$PYTHON $RUNNER --model=SASRec --dataset=Beauty_and_Personal_Care --config_files=$CONFIG/beauty_m1_fast.yaml

# M2 SASRecF
$PYTHON $RUNNER --model=SASRecF --dataset=Beauty_and_Personal_Care --config_files=$CONFIG/beauty_m2_fast.yaml

# M3 GMUSASRec
$PYTHON $RUNNER --model=GMUSASRec --dataset=Beauty_and_Personal_Care --config_files=$CONFIG/beauty_m3_fast.yaml
```

### Clothing_Shoes_and_Jewelry (MAX_ITEM_LIST_LENGTH=20, stopping_step=3)

```bash
# M1 SASRec
$PYTHON $RUNNER --model=SASRec --dataset=Clothing_Shoes_and_Jewelry --config_files=$CONFIG/clothing_m1_fast.yaml

# M2 SASRecF
$PYTHON $RUNNER --model=SASRecF --dataset=Clothing_Shoes_and_Jewelry --config_files=$CONFIG/clothing_m2_fast.yaml

# M3 GMUSASRec
$PYTHON $RUNNER --model=GMUSASRec --dataset=Clothing_Shoes_and_Jewelry --config_files=$CONFIG/clothing_m3_fast.yaml
```

## 결과 위치

- 로그: `RecBole/log/<MODEL>/<MODEL>-<DATASET>-<timestamp>.log`
- 체크포인트: `RecBole/saved/<MODEL>-<timestamp>.pth`

## 모델별 설정 요약

| 모델 | 설명 |
|---|---|
| SASRec (M1) | ID 임베딩 기반 sequential 추천 (baseline) |
| SASRecF (M2) | SASRec + 이미지 피처 concat |
| GMUSASRec (M3) | SASRec + 이미지 피처 GMU 융합 |

M3 전용 설정 (`*_m3_fast.yaml`):
- `numerical_features: [img_feat]` — FLOAT_SEQ를 (값, mask) 튜플로 적재
- `normalize_field: [img_feat]` — tanh 포화 방지용 정규화
- `discretization`에 `img_feat` 미포함 필수 (포함 시 fusion 파괴)
