import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100

print("\n[1단계] 실제 데이터 로드 및 전처리")
print("-" * 100)

data_files = [
    '구급 현황_2023_전국.csv',
    '구급 현황_2022_전국.csv',
    '구급 현황_2021_전국.csv',
    '구급 현황_2020_전국.csv',
    '구급 현황_2019_전국.csv'
]

# 데이터 로드
df_list = []
for file in data_files:
    try:
        print(f"로딩 중 : {file}...")
        df_temp = pd.read_csv(f"datas/{file}", encoding='utf-8', low_memory=False)
        df_temp.columns = df_temp.columns.str.lower()
        df_list.append(df_temp)
        print(f"  ✓ 완료 : {len(df_temp):,}건")
    except FileNotFoundError:
        print(f"  ✗ 파일을 찾을 수 없습니다 : {file}")
    except Exception as e:
        print(f"  ✗ 오류 발생 : {e}")

df = pd.concat(df_list, ignore_index=True)
print(f"\n✓ 전체 데이터 등록 완료 : {len(df):,}건")

# 컬럼명을 소문자로 변환
df.columns = df.columns.str.lower()

# 날짜/시간 변환 (YMD와 TM 컬럼 사용)
print("날짜/시간 컬럼 생성 중...")

# YMD(날짜) + TM(시간) 형식으로 결합 (시간은 6자리로 패딩: HHMMSS)
# NaN 값 처리를 위해 fillna 사용
df['신고일시'] = pd.to_datetime(
    df['dclr_ymd'].fillna(0).astype(int).astype(str) + 
    df['dclr_tm'].fillna(0).astype(int).astype(str).str.zfill(6),
    format='%Y%m%d%H%M%S', errors='coerce'
)

df['출동일시'] = pd.to_datetime(
    df['dspt_ymd'].fillna(0).astype(int).astype(str) + 
    df['dspt_tm'].fillna(0).astype(int).astype(str).str.zfill(6),
    format='%Y%m%d%H%M%S', errors='coerce'
)

df['현장도착일시'] = pd.to_datetime(
    df['grnds_arvl_ymd'].fillna(0).astype(int).astype(str) + 
    df['grnds_arvl_tm'].fillna(0).astype(int).astype(str).str.zfill(6),
    format='%Y%m%d%H%M%S', errors='coerce'
)

print(f"✓ 신고일시 유효 데이터: {df['신고일시'].notna().sum():,}건")
print(f"✓ 출동일시 유효 데이터: {df['출동일시'].notna().sum():,}건")
print(f"✓ 현장도착일시 유효 데이터: {df['현장도착일시'].notna().sum():,}건")

# 응답시간 및 서비스 지표 계산
df['응답시간_분'] = (df['출동일시'] - df['신고일시']).dt.total_seconds() / 60
df['현장도착시간_분'] = (df['현장도착일시'] - df['신고일시']).dt.total_seconds() / 60
df['현장처치시간_분'] = (df['현장도착일시'] - df['출동일시']).dt.total_seconds() / 60

# 이상치 제거
initial_count = len(df)
print(f"\n이상치 제거 전: {initial_count:,}건")

# 날짜/시간 결측치 제거
df = df.dropna(subset=['신고일시', '출동일시', '현장도착일시'])
print(f"✓ 날짜/시간 결측치 제거 후: {len(df):,}건")

# 응답시간 이상치 제거 (0~120분)
df = df[(df['응답시간_분'] >= 0) & (df['응답시간_분'] <= 120)]
print(f"✓ 응답시간 이상치 제거 후: {len(df):,}건")

# 현장도착시간 이상치 제거 (0~120분)
df = df[(df['현장도착시간_분'] >= 0) & (df['현장도착시간_분'] <= 120)]
print(f"✓ 현장도착시간 이상치 제거 후: {len(df):,}건")

print(f"\n총 제거된 데이터: {initial_count - len(df):,}건")

# 파생변수 생성
df['신고연도'] = df['신고일시'].dt.year
df['신고월'] = df['신고일시'].dt.month
df['신고일'] = df['신고일시'].dt.day
df['신고시간'] = df['신고일시'].dt.hour
df['신고분'] = df['신고일시'].dt.minute
df['신고요일'] = df['신고일시'].dt.day_name()
df['신고주차'] = df['신고일시'].dt.isocalendar().week

# 요일 한글 변환
weekday_dict = {
    'Monday': '월', 'Tuesday': '화', 'Wednesday': '수', 
    'Thursday': '목', 'Friday': '금', 'Saturday': '토', 'Sunday': '일'
}
df['신고요일'] = df['신고요일'].map(weekday_dict)

# 주말 여부
df['주말여부'] = df['신고일시'].dt.dayofweek.isin([5, 6]).astype(int)

# 계절
df['계절'] = df['신고월'].apply(lambda x: 
    '봄' if 3 <= x <= 5 else 
    '여름' if 6 <= x <= 8 else 
    '가을' if 9 <= x <= 11 else '겨울')

# 시간대 구분
df['시간대'] = pd.cut(df['신고시간'], 
                      bins=[-1, 6, 12, 18, 24], 
                      labels=['새벽(0-6)', '오전(6-12)', '오후(12-18)', '저녁(18-24)'])

# 출퇴근시간 여부
df['출퇴근시간'] = df['신고시간'].isin([7, 8, 9, 17, 18, 19]).astype(int)

# 컬럼명 매핑
column_mapping = {
    'ctpv_nm': '시도명',
    'gndr_nm': '성별',
    'ptn_ocrn_type_nm': '발생유형',
    'grnds_dstnc': '현장거리',
    'cty_frmvl_se_nm': '도시농촌',
    'hrtarst_nm': '심정지',
    'sril_oncr_nm': '중증외상',
}

for old_col, new_col in column_mapping.items():
    if old_col in df.columns:
        df[new_col] = df[old_col]

print(f"\n✓ 전처리 완료 : {len(df):,}건")
print(f"총 컬럼 수 : {len(df.columns)}개")

print("\n[2단계] 시계열 분해 및 트렌드 분석")
print("-" * 100)

# 일별 집계
daily_counts = df.groupby(df['신고일시'].dt.date).size()
daily_counts.index = pd.to_datetime(daily_counts.index)
daily_counts = daily_counts.sort_index()

# 시계열 분해 (추세, 계절성, 잔차)
if len(daily_counts) > 365:
    decomposition = seasonal_decompose(daily_counts, model='additive', period=365)
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # 원본 데이터
    axes[0].plot(daily_counts.index, daily_counts.values, color='steelblue', linewidth=1.5)
    axes[0].set_title('일별 구급 출동 건수 (원본)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('출동 건수')
    axes[0].grid(alpha=0.3)
    
    # 추세
    axes[1].plot(decomposition.trend.index, decomposition.trend.values, 
                 color='darkred', linewidth=2)
    axes[1].set_title('추세', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('추세')
    axes[1].grid(alpha=0.3)
    
    # 계절성
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 
                 color='darkgreen', linewidth=1)
    axes[2].set_title('계절성', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('계절성')
    axes[2].grid(alpha=0.3)
    
    # 잔차
    axes[3].plot(decomposition.resid.index, decomposition.resid.values, 
                 color='darkorange', linewidth=0.5, alpha=0.7)
    axes[3].set_title('잔차', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('잔차')
    axes[3].set_xlabel('날짜')
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timeseries_decomposition.png', dpi=300, bbox_inches='tight')
    print("✓ timeseries_decomposition.png")
    plt.close()
    
    # 자기상관 분석
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    plot_acf(daily_counts.dropna(), lags=50, ax=axes[0])
    axes[0].set_title('자기상관함수 (ACF)', fontsize=14, fontweight='bold')
    
    plot_pacf(daily_counts.dropna(), lags=50, ax=axes[1])
    axes[1].set_title('부분자기상관함수 (PACF)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('autocorrelation.png', dpi=300, bbox_inches='tight')
    print("✓ autocorrelation.png")
    plt.close()

print("\n[3단계] 클러스터링 분석")
print("-" * 100)

# 시도명 컬럼 확인 및 처리
region_col = None
if '시도명' in df.columns:
    region_col = '시도명'
elif 'ctpv_nm' in df.columns:
    df['시도명'] = df['ctpv_nm']
    region_col = '시도명'
else:
    print("⚠️ 지역 정보를 찾을 수 없습니다. 클러스터링을 건너뜁니다.")
    region_col = None

if region_col is not None:
    # 지역별 특성 추출
    region_features = df.groupby(region_col).agg({
        '응답시간_분': ['mean', 'std'],
        '현장도착시간_분': 'mean',
        '현장거리': 'mean',
        '신고일시': 'count',
        '주말여부': 'mean',
        '출퇴근시간': 'mean'
    }).reset_index()
    
    region_features.columns = ['시도명', '평균응답시간', '응답시간표준편차', 
                              '평균현장도착시간', '평균거리', '출동건수', 
                              '주말비율', '출퇴근시간비율']
    
    # 결측치 제거
    region_features = region_features.dropna()
    
    print(f"✓ 분석 대상 지역 수: {len(region_features)}개")
    
    if len(region_features) > 0:
        # 표준화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(
            region_features[['평균응답시간', '응답시간표준편차', '평균현장도착시간', 
                             '평균거리', '출동건수', '주말비율', '출퇴근시간비율']]
        )
        
        # 클러스터 수 결정 (지역 수에 따라 조정)
        n_clusters = min(4, len(region_features))
        
        # K-Means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        region_features['클러스터'] = kmeans.fit_predict(features_scaled)
        
        # PCA로 2차원 시각화
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # K-Means 결과
        scatter = axes[0].scatter(features_pca[:, 0], features_pca[:, 1], 
                                 c=region_features['클러스터'], 
                                 cmap='viridis', s=200, alpha=0.7, edgecolors='black')
        for i, region in enumerate(region_features['시도명']):
            axes[0].annotate(region, (features_pca[i, 0], features_pca[i, 1]), 
                            fontsize=9, ha='center')
        axes[0].set_title('지역 클러스터링 (K-Means, PCA)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 설명력)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 설명력)')
        plt.colorbar(scatter, ax=axes[0], label='클러스터')
        axes[0].grid(alpha=0.3)
        
        # 계층적 클러스터링
        linkage_matrix = linkage(features_scaled, method='ward')
        dendrogram(linkage_matrix, labels=region_features['시도명'].values, 
                  ax=axes[1], leaf_font_size=10)
        axes[1].set_title('계층적 클러스터링 덴드로그램', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('지역')
        axes[1].set_ylabel('거리')
        
        plt.tight_layout()
        plt.savefig('clustering.png', dpi=300, bbox_inches='tight')
        print("✓ clustering.png")
        plt.close()
        
        # 클러스터별 특성 분석
        print("\n클러스터별 특성:")
        for cluster in sorted(region_features['클러스터'].unique()):
            cluster_regions = region_features[region_features['클러스터'] == cluster]
            print(f"\n클러스터 {cluster} : {', '.join(cluster_regions['시도명'].values)}")
            print(f"  평균 응답시간 : {cluster_regions['평균응답시간'].mean():.2f}분")
            print(f"  평균 출동건수 : {cluster_regions['출동건수'].mean():.0f}건")
    else:
        print("⚠️ 분석할 수 있는 지역 데이터가 없습니다.")
else:
    print("⚠️ 클러스터링 분석을 건너뛰었습니다.")

print("\n[4단계] 머신러닝 예측 모델")
print("-" * 100)

# 특성 공학
ml_df = df.copy()

# 범주형 변수 인코딩
label_encoders = {}
categorical_cols = ['시도명', '성별', '발생유형', '도시농촌', '시간대', '계절', '신고요일']

for col in categorical_cols:
    if col in ml_df.columns:
        le = LabelEncoder()
        ml_df[col + '_encoded'] = le.fit_transform(ml_df[col].astype(str))
        label_encoders[col] = le

# 특성 선택
feature_cols = ['신고시간', '신고월', '신고일', '현장거리', '주말여부', 
                '출퇴근시간'] + [col + '_encoded' for col in categorical_cols 
                if col in ml_df.columns]

# 결측치 제거
ml_df = ml_df[feature_cols + ['응답시간_분']].dropna()

# 대용량 데이터이므로 샘플링 (100만 건으로 제한)
if len(ml_df) > 1000000:
    print(f"  데이터 샘플링 : {len(ml_df):,}건 -> 1,000,000건")
    ml_df = ml_df.sample(n=1000000, random_state=42)

X = ml_df[feature_cols]
y = ml_df['응답시간_분']

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  학습 데이터 : {len(X_train):,}건")
print(f"  테스트 데이터 : {len(X_test):,}건")

# 모델 학습 (빠른 실행을 위해 파라미터 조정)
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, 
                                          random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, max_depth=5, 
                                                   random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    # 평가
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    print(f"  RMSE : {rmse:.4f}분")
    print(f"  MAE : {mae:.4f}분")
    print(f"  R² : {r2:.4f}")

# 결과 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 모델 성능 비교
metrics_df = pd.DataFrame(results).T
axes[0, 0].bar(metrics_df.index, metrics_df['R2'], color=['steelblue', 'coral', 'lightgreen'])
axes[0, 0].set_title('모델별 R² 스코어', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(metrics_df['R2']):
    axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')

# 예측 vs 실제 (Random Forest)
best_model_name = max(results, key=lambda x: results[x]['R2'])
best_predictions = predictions[best_model_name]
axes[0, 1].scatter(y_test, best_predictions, alpha=0.3, s=10)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
               'r--', linewidth=2, label='완벽한 예측')
axes[0, 1].set_title(f'{best_model_name}: 예측 vs 실제', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('실제 응답시간 (분)')
axes[0, 1].set_ylabel('예측 응답시간 (분)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 특성 중요도 (Random Forest)
if best_model_name == 'Random Forest':
    feature_importance = pd.DataFrame({
        '특성': feature_cols,
        '중요도': models[best_model_name].feature_importances_
    }).sort_values('중요도', ascending=False).head(10)
    
    axes[1, 0].barh(range(len(feature_importance)), feature_importance['중요도'], 
                   color='teal', alpha=0.7)
    axes[1, 0].set_yticks(range(len(feature_importance)))
    axes[1, 0].set_yticklabels(feature_importance['특성'])
    axes[1, 0].set_title('특성 중요도 (상위 10개)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('중요도')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].invert_yaxis()

# 잔차 분석
residuals = y_test - best_predictions
axes[1, 1].scatter(best_predictions, residuals, alpha=0.3, s=10)
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_title('잔차 플롯', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('예측값')
axes[1, 1].set_ylabel('잔차')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('machine_learning.png', dpi=300, bbox_inches='tight')
print("\n✓ machine_learning.png")
plt.close()

print("\n[5단계] 고급 통계 분석")
print("-" * 100)

# 다중 회귀 분석
from scipy.stats import shapiro, anderson
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# ANOVA - 시간대별 응답시간 차이
time_groups = [df[df['시간대'] == time]['응답시간_분'].dropna() 
               for time in df['시간대'].unique() if pd.notna(time)]
f_stat, p_value = stats.f_oneway(*time_groups)
print(f"\nANOVA 분석 (시간대별 응답시간) :")
print(f"  F-통계량 : {f_stat:.4f}")
print(f"  p-value : {p_value:.6f}")
print(f"  결론 : {'시간대별 유의미한 차이 존재' if p_value < 0.05 else '시간대별 차이 없음'}")

# 카이제곱 검정 - 계절과 발생유형의 독립성
if '계절' in df.columns and '발생유형' in df.columns:
    contingency_table = pd.crosstab(df['계절'], df['발생유형'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\n카이제곱 검정 (계절 vs 발생유형) :")
    print(f"  χ² 통계량 : {chi2:.4f}")
    print(f"  p-value : {p_value:.6f}")
    print(f"  자유도 : {dof}")
    print(f"  결론 : {'계절과 발생유형은 관련이 있음' if p_value < 0.05 else '독립적'}")

# 상관관계 히트맵 (확장)
numeric_cols = ['응답시간_분', '현장도착시간_분', '현장처치시간_분', 
                '현장거리', '신고시간', '신고월', '주말여부', '출퇴근시간']
numeric_cols = [col for col in numeric_cols if col in df.columns]

correlation_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.3f', 
            cmap='coolwarm', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('확장 상관관계 히트맵', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n✓ correlation_heatmap.png")
plt.close()

print("\n[6단계] 이상치 탐지")
print("-" * 100)

from sklearn.ensemble import IsolationForest

# Isolation Forest
iso_features = df[['응답시간_분', '현장도착시간_분', '현장거리']].dropna()
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(iso_features)

df_iso = iso_features.copy()
df_iso['이상치'] = outliers

# Z-Score 방법
z_scores = np.abs(stats.zscore(iso_features))
z_outliers = (z_scores > 3).any(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Isolation Forest 결과
axes[0].scatter(df_iso[df_iso['이상치'] == 1]['응답시간_분'], 
               df_iso[df_iso['이상치'] == 1]['현장도착시간_분'], 
               c='blue', alpha=0.5, s=10, label='정상')
axes[0].scatter(df_iso[df_iso['이상치'] == -1]['응답시간_분'], 
               df_iso[df_iso['이상치'] == -1]['현장도착시간_분'], 
               c='red', alpha=0.8, s=30, label='이상치', marker='x')
axes[0].set_title('Isolation Forest 이상치 탐지', fontsize=14, fontweight='bold')
axes[0].set_xlabel('응답시간 (분)')
axes[0].set_ylabel('현장도착시간 (분)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Z-Score 결과
axes[1].scatter(iso_features[~z_outliers]['응답시간_분'], 
               iso_features[~z_outliers]['현장도착시간_분'], 
               c='blue', alpha=0.5, s=10, label='정상')
axes[1].scatter(iso_features[z_outliers]['응답시간_분'], 
               iso_features[z_outliers]['현장도착시간_분'], 
               c='red', alpha=0.8, s=30, label='이상치', marker='x')
axes[1].set_title('Z-Score 이상치 탐지 (Z>3)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('응답시간 (분)')
axes[1].set_ylabel('현장도착시간 (분)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
print("✓ outlier_detection.png")
plt.close()

print(f"\nIsolation Forest 탐지 결과 :")
print(f"  이상치 수 : {(outliers == -1).sum():,}건 ({(outliers == -1).sum() / len(outliers) * 100:.2f}%)")
print(f"\nZ-Score 탐지 결과 :")
print(f"  이상치 수 : {z_outliers.sum():,}건 ({z_outliers.sum() / len(z_outliers) * 100:.2f}%)")

print("\n[7단계] 지역 간 상호작용 네트워크 분석")
print("-" * 100)

# 시도별 이송 패턴 분석
if '시도명' in df.columns:
    # 시간대별 지역 출동 패턴
    time_region_pattern = pd.crosstab(df['시간대'], df['시도명'], normalize='columns') * 100
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(time_region_pattern, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': '비율 (%)'}, ax=ax)
    ax.set_title('시간대별 × 지역별 출동 패턴 (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('시도')
    ax.set_ylabel('시간대')
    plt.tight_layout()
    plt.savefig('time_region_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ time_region_heatmap.png")
    plt.close()

print("\n[8단계] 종합 대시보드 생성")
print("-" * 100)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 연도별 추이
ax1 = fig.add_subplot(gs[0, :2])
yearly_stats = df.groupby('신고연도').agg({
    '신고일시': 'count',
    '응답시간_분': 'mean'
}).reset_index()
ax1_twin = ax1.twinx()
ax1.bar(yearly_stats['신고연도'], yearly_stats['신고일시'], 
        color='steelblue', alpha=0.7, label='출동 건수')
ax1_twin.plot(yearly_stats['신고연도'], yearly_stats['응답시간_분'], 
             color='red', marker='o', linewidth=2, markersize=8, label='평균 응답시간')
ax1.set_xlabel('연도')
ax1.set_ylabel('출동 건수', color='steelblue')
ax1_twin.set_ylabel('평균 응답시간 (분)', color='red')
ax1.set_title('연도별 출동 건수 및 평균 응답시간', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(alpha=0.3)

# 2. 월별 박스플롯
ax2 = fig.add_subplot(gs[0, 2])
monthly_data = [df[df['신고월'] == m]['응답시간_분'].dropna() for m in range(1, 13)]
bp = ax2.boxplot(monthly_data, labels=range(1, 13), patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax2.set_title('월별 응답시간 분포', fontsize=12, fontweight='bold')
ax2.set_xlabel('월')
ax2.set_ylabel('응답시간 (분)')
ax2.grid(axis='y', alpha=0.3)

# 3. 요일-시간 히트맵
ax3 = fig.add_subplot(gs[1, :])
weekday_order = ['월', '화', '수', '목', '금', '토', '일']
pivot_data = df.groupby(['신고요일', '신고시간']).size().unstack(fill_value=0)
pivot_data = pivot_data.reindex(weekday_order)
sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': '출동 건수'}, ax=ax3)
ax3.set_title('요일 × 시간대 출동 히트맵', fontsize=12, fontweight='bold')
ax3.set_xlabel('시간')
ax3.set_ylabel('요일')

# 4. 지역별 Top 10
ax4 = fig.add_subplot(gs[2, 0])
if '시도명' in df.columns:
    top_regions = df['시도명'].value_counts().head(10)
    ax4.barh(range(len(top_regions)), top_regions.values, color='coral')
    ax4.set_yticks(range(len(top_regions)))
    ax4.set_yticklabels(top_regions.index)
    ax4.set_title('지역별 출동 건수 Top 10', fontsize=12, fontweight='bold')
    ax4.set_xlabel('출동 건수')
    ax4.grid(axis='x', alpha=0.3)
    ax4.invert_yaxis()

# 5. 발생유형 파이차트
ax5 = fig.add_subplot(gs[2, 1])
if '발생유형' in df.columns:
    occurrence_counts = df['발생유형'].value_counts()
    ax5.pie(occurrence_counts.values, labels=occurrence_counts.index, 
           autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set3'))
    ax5.set_title('발생 유형별 분포', fontsize=12, fontweight='bold')

# 6. 응답시간 히스토그램
ax6 = fig.add_subplot(gs[2, 2])
ax6.hist(df['응답시간_분'].dropna(), bins=50, color='lightgreen', 
        edgecolor='black', alpha=0.7)
ax6.axvline(df['응답시간_분'].mean(), color='red', linestyle='--', 
           linewidth=2, label=f"평균: {df['응답시간_분'].mean():.1f}분")
ax6.axvline(df['응답시간_분'].median(), color='blue', linestyle='--', 
           linewidth=2, label=f"중앙값: {df['응답시간_분'].median():.1f}분")
ax6.set_title('응답시간 분포', fontsize=12, fontweight='bold')
ax6.set_xlabel('응답시간 (분)')
ax6.set_ylabel('빈도')
ax6.legend()
ax6.grid(alpha=0.3)

plt.suptitle('소방청 구급 현황 종합 대시보드', fontsize=18, fontweight='bold', y=0.995)
plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ comprehensive_dashboard.png")
plt.close()

print("\n[9단계] 분석 결과 요약")
print("=" * 100)

summary = {
    '데이터 개요': {
        '총 데이터 건수': f"{len(df):,}건",
        '분석 기간': f"{df['신고연도'].min()}년 ~ {df['신고연도'].max()}년",
        '분석 컬럼 수': len(df.columns)
    },
    '기술 통계': {
        '평균 응답시간': f"{df['응답시간_분'].mean():.2f}분",
        '중앙값 응답시간': f"{df['응답시간_분'].median():.2f}분",
        '평균 현장도착시간': f"{df['현장도착시간_분'].mean():.2f}분",
        '골든타임 달성률': f"{(df['현장도착시간_분'] <= 10).mean() * 100:.2f}%"
    },
    '머신러닝 성능': {
        '최고 성능 모델': best_model_name,
        'R² Score': f"{results[best_model_name]['R2']:.4f}",
        'RMSE': f"{results[best_model_name]['RMSE']:.4f}분",
        'MAE': f"{results[best_model_name]['MAE']:.4f}분"
    },
    '생성된 시각화': [
        'timeseries_decomposition.png',
        'autocorrelation.png',
        'clustering.png',
        'machine_learning.png',
        'correlation_heatmap.png',
        'outlier_detection.png',
        'time_region_heatmap.png',
        'comprehensive_dashboard.png'
    ]
}

print("\n분석 요약 :")
for category, items in summary.items():
    print(f"\n■ {category}")
    if isinstance(items, dict):
        for key, value in items.items():
            print(f"  • {key}: {value}")
    elif isinstance(items, list):
        for item in items:
            print(f"  • {item}")

# JSON 저장
import json
with open('analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 100)
print("✓ 분석 완료")
print(f"✓ 총 {len(summary['생성된 시각화'])}개의 시각화 생성")
print("✓ 분석 요 약: analysis_summary.json")
print("=" * 100)
