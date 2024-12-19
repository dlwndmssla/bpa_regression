import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler


path = 'C:/Users/user/OneDrive - pusan.ac.kr/바탕 화면/부산항만공사 물동량 예측/'
# st.set_page_config(layout="wide")

##############################################
#              DEF. FUCNTION                #     
##############################################
def forward_selection(data, target, significance_level=0.1):

    initial_features = data.columns.tolist()
    selected_features = []
    while initial_features:
        remaining_features = list(set(initial_features) - set(selected_features))
        new_pval = pd.Series(index=remaining_features, dtype=float)
        
        for feature in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[selected_features + [feature]])).fit()
            new_pval[feature] = model.pvalues[feature]
        
        min_pval = new_pval.min()
        if min_pval < significance_level:
            best_feature = new_pval.idxmin()
            selected_features.append(best_feature)
            print(f"Selected feature: {best_feature} (p-value: {min_pval})")
        else:
            break

    final_model = sm.OLS(target, sm.add_constant(data[selected_features])).fit()
    return selected_features, final_model






# 제목 추가
st.title("부산항만공사 물동량 예측")

# 간단한 텍스트 출력
# st.write("부산대학교 산업공학과 석사과정 이주은")
st.write("목표: 부산항만공사의 년단위 컨테이너 물동량(TEU)를 예측하고자 함")
st.write("컨테이너 물동량의 예측은 회귀분석을 통해 수행할 예정이며 모델은 선형회귀이다.")
st.write("학습데이터는 부산항만공사의 년단위 물동량 데이터가 존재하는 2010년부터 2023년까지이며")
st.write("예측은 2024년부터 2029년에 대해 수행한다.")



st.write("---")
st.markdown("")
st.markdown("### 부산항만공사 물동량 예측 순서")
st.markdown("1. 데이터 수집")
st.markdown("2. 대한민국 수출입 상위 20개국 추출")
st.markdown("3. 수출입 상위 20개국을 기준으로 WEO 데이터의 국가 선별")
st.markdown("4. 선별된 WEO 데이터를 이용한 변수 선택  -  forward selection")
st.markdown("5. 선택된 변수를 이용한 년도별(2024 ~ 2029년) 부산항 물동량 예측")

st.write("---")
st.write("### 1. 데이터 수집")

st.markdown("[k-stat(국가-지역별 수출입)](https://stat.kita.net/stat/world/trade/CtrProdImpExpList.screen)")
st.write("- 대한민국 수출입 상위 20개국 추출에 사용")
st.write("")
st.markdown("[World Economic Outlook database](https://www.imf.org/en/Publications/WEO/weo-database/2023/October/download-entire-database)")
st.write("- 국가별 경제지표 데이터  (by countries (9MB)) ")
st.write("")
st.markdown("[부산항 과거 물동량 데이터](https://www.busanpa.com/kor/Contents.do?mCode=MN1003)")
st.write("- 2010년부터 2023년의 년간 컨테이너물동량 데이터")
st.write("")


ARIMA = [24354000,25238000,26372000,27470000]
ARIMA_X = [25563000,26863000,28197000,29581000]

st.write("---")
st.write("### 2. 대한민국 수출입 상위 20개국 추출")
st.write("기준 년도 : 2023년, 금액단위 : 백만불")
trade_list = pd.DataFrame(pd.read_excel(path+"K-stat 수출입 무역통계_2023.xlsx"))

st.table(trade_list)
##############################################
#                DATA INPUT                 #     
##############################################

# 샘플 데이터 생성 (변수가 5개인 100개의 샘플)
data = pd.DataFrame(pd.read_excel(path+'pca/total_container.xlsx'))
data_col = pd.DataFrame(pd.read_excel(path+'pca/WEOOct2024all.xlsx'))
data_col['key'] =  data_col['WEO Subject Code']+'_'+data_col['Country']
col_list = ['key', 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

pca_data = data_col[col_list].iloc[:,1:].apply(pd.to_numeric, errors="coerce").T
pca_data.columns = data_col['key']
pca_data = pca_data.dropna(axis=1).reset_index(drop=True)

top20_iso =  ['VNM', 'SGP', 'TUR', 'DEU', 'IND', 'CHN', 'HUN', 'IDN', 'AUS', 'KOR', 'NLD', 'ITA', 'POL', 'MYS', 'USA', 'GBR', 'THA', 'HKG', 'TWN', 'JPN', 'CAN', 'MEX', 'PHL', 'RUS']

# ['KOR','CHN', 'VNM', 'GBR', 'PHL', 'IDN', 'ITA', 'CAN', 'NLD', 'MEX', 'DEU', 'USA', 'SGP', 'TWN', 'HKG', 'MYS', 'THA', 'JPN', 'IND', 'AUS', 'RUS']
# top20_iso =  ['KOR',"CHN","USA","VNM","JPN","HKG"]

data_col.loc[data_col['ISO'].isin(top20_iso)].iloc[:,[0,1,2,3,]+[i for i in range(-8-14,-2)]].to_csv('top20_10years.csv')
data = pd.DataFrame(pd.read_csv(path+'top20_10years.csv'))

data['reg_col'] = [data['WEO Subject Code'][i]+f'({data.ISO[i]})' for i in range(len(data))]
df0 = data.iloc[:,5:].iloc[:,::-1].T
df0.columns = df0.iloc[0]
df0 = df0.iloc[1:]


# container = [0,3172000,19469000,19456000,20493000,21663000,21992000,21824000,22706000,22078000,23154000][::-1]
container = [14158211, 16076548, 17046177, 17686099, 18683283, 19468725, 19456291, 20493475, 21662572, 21992000, 21823995, 22706130, 22078195, 23153508,0,1,2,3,4,5][::-1]
# for i in list(df.index):
#     df2 = pd.DataFrame(pd.read_excel(f'pca/월별 화물 처리실적(확정)_{i}.xlsx'))
#     print(i,df2.iloc[-1,1])
#     container.append(df2.iloc[-1,1])

df0['container'] = container
df0.to_csv('container_13-23.csv')




# Feature, target 분리
data = pd.DataFrame(pd.read_csv(path+'container_13-23.csv')).iloc[::-1].reset_index(drop=True)
data = data.rename(columns={'Unnamed: 0':'year'})

df = data.drop(columns='container').drop(columns='year')
target = data['container']

ax = pd.DataFrame(df.describe().iloc[0] != len(df))
bx = list(ax[ax['count']].T.columns)
cx = []
for i in bx:
    # print(i[:i.index("(")])
    cx.append(i[:i.index("(")])
cx = set(cx)

dx = []
for i in list(data.columns)[1:-1]:
    if 'KOR' in i: dx.append(i)
    elif i[:i.index("(")] in cx:
        # print(i[:i.index("(")])
        continue
    else: dx.append(i)


# Feature, target 분리
data = pd.DataFrame(pd.read_csv(path+'container_13-23.csv')).iloc[::-1].reset_index(drop=True)
data = data.rename(columns={'Unnamed: 0':'year'})

df = data[['year']+dx]
print(df.columns[df.isna().any()])
df = df.dropna(axis=1)
print(df.columns[df.isna().any()])
# df = data[['year']+data.filter(like='KOR', axis=1).columns.tolist()]
target = data['container']

# train, test 데이터 분리 (8 : 2)
# X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2)
X_train, X_test, y_train, y_test = df.iloc[:14],df.iloc[14:],target.iloc[:14],target.iloc[14:]
# 연료 변수가 object이므로 더미화 진행 (type : 3개)


st.write("---")
st.write("### 3.수출입 상위 20개국을 기준 WEO 데이터의 국가 선별")
st.table(df)


##############################################
#            Perform forward selection       #     
##############################################
# selected_features, final_model = forward_selection(X_train, y_train)
# print("\nSelected Features:", selected_features)
# print("\nFinal Model Summary:")
# print(final_model.summary())



##############################################
#             LINEAR REGRESSION              #     
##############################################
###############################
#       SELECTED KOREA        #     
###############################
st.write("---")
st.write("### 4. 선별된 WEO 데이터를 이용한 변수 선택")
st.write("#### -  forward selection")
a = ['PCPIEPCH(IDN)', 'PPPEX(USA)', 'PCPIPCH(IND)', 'NID_NGDP(NLD)', 'PCPIPCH(VNM)', 'PPPEX(PHL)', 'PPPSH(PHL)', 'NGDP_RPCH(IDN)']
b = ['NGSD_NGDP(JPN)', 'PPPEX(USA)', 'TX_RPCH(USA)', 'PCPIPCH(VNM)']
c = ['PPPEX(USA)', 'NGDPDPC(HKG)', 'BCA(MYS)', 'GGR_NGDP(DEU)', 'GGX_NGDP(THA)', 'GGXCNL_NGDP(VNM)', 'GGXWDG_NGDP(CHN)']
d = ['PPPEX(USA)', 'NGDP_D(HKG)', 'NGDPRPPPPC(VNM)', 'TXG_RPCH(JPN)', 'TX_RPCH(JPN)', 'NGDP_R(VNM)']


print(y_test)


# 기존 플롯
plt.figure(figsize=(12, 6))
plt.plot(X_train['year'], y_train, color='black', label='Train Data')
plt.xlabel("Year")
plt.ylabel("Container")
pred_list = []
# 모델 학습 및 결과 플롯
for k, reg_col in enumerate([a, b, c, d]):
    kor = [i for i in dx if 'KOR' in i]
    reg = ['year']
    reg_set = set([i[:i.index("(")] for i in reg_col])  # reg_col의 '(' 이전까지 추출
    st.write(f"{k+1}변수 조합",{'year'},'+',reg_set)
    print(reg_set, len(reg_set))
    
    for j in df.columns[1:]:
        if j[:j.index("(")] in reg_set:  # reg_set에 포함되면 추가
            reg.append(j)
    
    model = LinearRegression()
    print('len(col)', len(reg) - len(kor), len(kor))
    model.fit(X_train[reg], y_train)

    # 예측
    y_pred = model.predict(X_test[reg])
    pred_list.append(y_pred)
    plt.plot(
        X_test['year'], y_pred,
        label=f"Prediction {k+1}",
        linestyle='-', 
        marker=['o', 's', '*', '^'][k],  # 마커 스타일
        markersize=5
    )

    # 모델 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print()

plt.plot([2024,2025,2026,2027], ARIMA,    label=f"bpa_arima",    linestyle='-',linewidth=3)
plt.plot([2024,2025,2026,2027], ARIMA_X,    label=f"bpa_arima_x",    linestyle='-', linewidth=3)
st.write('---')
# 그래프 설정
plt.xticks(data['year'])
plt.grid()
plt.ylim([1e7, 4.5e7])
plt.legend()
plt.tight_layout()
st.write("### 5.1 시나리오 1")
st.write("#### 대한민국 경제지표 - forward selection과 동일 선택")
st.pyplot(plt)
st.write("")
st.write("시나리오별 컨테이너 물동량 예측치 - 단위 TEU")
pred_list = [list(map(int,i)) for i in pred_list]
pred_list = pd.DataFrame(pred_list,columns=[2024,2025,2026,2027,2028,2029])
pred_list.index += 1
st.table(pred_list)
st.write("시나리오별 변수")
for k, reg_col in enumerate([a, b, c, d]):
    reg_set = set([i[:i.index("(")] for i in reg_col])
    st.write(f"시나리오{k+1}:",{'year'},'+ 국가별',reg_set)




###############################
#          FULL KOREA         #     
###############################
st.markdown('---') 

a = ['PCPIEPCH(IDN)', 'PPPEX(USA)', 'PCPIPCH(IND)', 'NID_NGDP(NLD)', 'PCPIPCH(VNM)', 'PPPEX(PHL)', 'PPPSH(PHL)', 'NGDP_RPCH(IDN)']
b = ['NGSD_NGDP(JPN)', 'PPPEX(USA)', 'TX_RPCH(USA)', 'PCPIPCH(VNM)']
c = ['PPPEX(USA)', 'NGDPDPC(HKG)', 'BCA(MYS)', 'GGR_NGDP(DEU)', 'GGX_NGDP(THA)', 'GGXCNL_NGDP(VNM)', 'GGXWDG_NGDP(CHN)']
d = ['PPPEX(USA)', 'NGDP_D(HKG)', 'NGDPRPPPPC(VNM)', 'TXG_RPCH(JPN)', 'TX_RPCH(JPN)', 'NGDP_R(VNM)']


plt.figure(figsize=(12, 6))
plt.plot(X_train['year'], y_train, color='black', label='Train Data')
plt.xlabel("Year")
plt.ylabel("Container")
pred_list = []

# 모델 학습 및 결과 플롯
for k, reg_col in enumerate([a, b, c, d]):
    kor = [i for i in dx if 'KOR' in i]
    reg = ['year']
    reg_set = set([i[:i.index("(")] for i in reg_col])  # reg_col의 '(' 이전까지 추출
    print(reg_set, len(reg_set))
    
    for j in df.columns[1:]:
        if j[:j.index("(")] in reg_set or j in kor:  # reg_set에 포함되면 추가
            reg.append(j)
    
    model = LinearRegression()
    print('len(col)', len(reg) - len(kor), len(kor))
    model.fit(X_train[reg], y_train)

    # 예측
    y_pred = model.predict(X_test[reg])
    pred_list.append(y_pred)
    plt.plot(
        X_test['year'], y_pred,
        label=f"Prediction {k+1}",
        linestyle='-', 
        marker=['o', 's', '*', '^'][k],  # 마커 스타일
        markersize=5
    )

    # 모델 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print()

# 그래프 설정
plt.plot([2024,2025,2026,2027], ARIMA,    label=f"bpa_arima",    linestyle='-',linewidth=3)
plt.plot([2024,2025,2026,2027], ARIMA_X,    label=f"bpa_arima_x",    linestyle='-', linewidth=3)
plt.xticks(data['year'])
plt.grid()
plt.ylim([1e7, 4.5e7])
plt.legend()
plt.tight_layout()
st.write("### 5.2  시나리오 2")
st.write("#### 대한민국 경제지표 - forward selection과 별개로 전체 선택")
st.pyplot(plt)
st.write("")
st.write("시나리오별 컨테이너 물동량 예측치 - 단위 TEU")
pred_list = [list(map(int,i)) for i in pred_list]
pred_list = pd.DataFrame(pred_list,columns=[2024,2025,2026,2027,2028,2029])
pred_list.index += 1
st.table(pred_list)
st.write("시나리오별 변수")
for k, reg_col in enumerate([a, b, c, d]):
    reg_set = set([i[:i.index("(")] for i in reg_col])
    st.write(f"시나리오{k+1}: ",{'year'},"+ 한국 전체 변수",'+ 국가별',reg_set)




def regression(cell_id):
    weo_col = pd.DataFrame(pd.read_csv(path+'weo.csv',encoding='cp949'))
    st.write("- 년도 / 국가 / WEO 변수 선택")
    # st.write(에 포함되어 있는 WEO 변수는 선택해도 반영되지 않습니다./t(해당 변수는 선택지의 끝부분에 NULL 이라고 표시되어 있습니다.).")
    st.write(f"- {cx}에 포함되어 있는 WEO 변수는 선택해도 반영되지 않습니다.")
    st.write("- (해당 변수는 선택지의 끝부분에 NULL 이라고 표시되어 있습니다.)")
    st.write("- 분석 반영 제외 년도 선택은 코로나19같은 event에 의해 영향을 받은것을 사전에 제외해보고자 추가하였습니다.")
    weo_col['x'] = weo_col['col']+' : '+weo_col['explanation']

    iso = {
        "한국": "KOR", "중국": "CHN", "미국": "USA", "베트남": "VNM", "일본": "JPN",
        "홍콩": "HKG", "대만": "TWN", "싱가포르": "SGP", "인도(인디아)": "IND", "호주": "AUS",
        "멕시코": "MEX", "독일": "DEU", "말레이시아": "MYS", "인도네시아": "IDN", "폴란드": "POL",
        "필리핀": "PHL", "튀르키예": "TUR", "캐나다": "CAN", "태국": "THA", "네덜란드": "NLD", "헝가리": "HUN"
    }
    top_country = ['한국', '중국', '미국', '베트남', '일본', '홍콩', '대만', '싱가포르', '인도(인디아)', '호주', 
                '멕시코', '독일', '말레이시아', '인도네시아', '폴란드', '필리핀', '튀르키예', '캐나다', '태국', 
                '네덜란드', '헝가리']
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

    # 3개의 열로 나누기
    col1, col2, col3 = st.columns([1, 1.5, 4.5])  # 비율: col1(2), col2(3), col3(1)

    # WEO 변수
    with col3:
        st.write("WEO 변수")
        weo_check = [st.checkbox(stock, key=str(cell_id)+stock) for stock in weo_col['x']]

    # 대한민국 수출입 상위 20개국
    with col2:
        st.write("대한민국 수출입 상위 20개국")
        country_check = [st.checkbox(stock, key=str(cell_id)+iso[stock]) for stock in top_country]

    # 분석 반영 년도 선택
    with col1:
        st.write("분석 반영 제외 년도 선택")
        years_check = [st.checkbox(str(years[i]), key=str(cell_id)+f"year_{i}") for i in range(len(years))]

    # 선택된 결과 출력 (선택된 부분에 따라 다르게 출력할 수 있음)


    selected_var = [weo_col['col'].iloc[i] for i in range(len(weo_check)) if weo_check[i] and weo_col['col'].iloc[i] not in cx]
    selected_nat = [iso[top_country[i]] for i in range(len(country_check)) if country_check[i]]
    selected_year = [year for year, checked in zip(years, years_check) if checked]
    total_var = ['year']+[f'{i}({j})' for i in selected_var for j in selected_nat]
    st.write("")
    st.write("")




    data = pd.DataFrame(pd.read_csv(path+'container_13-23.csv')).iloc[::-1].reset_index(drop=True)
    data = data.rename(columns={'Unnamed: 0':'year'})

    df = data[total_var]
    print(df.columns[df.isna().any()])
    df = df.dropna(axis=1)
    print(df.columns[df.isna().any()])
    # df = data[['year']+data.filter(like='KOR', axis=1).columns.tolist()]
    target = data['container']

    # train, test 데이터 분리 (8 : 2)
    # X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2)
    X_train, X_test, y_train, y_test = df.iloc[:14],df.iloc[14:],target.iloc[:14],target.iloc[14:]
    # 연료 변수가 object이므로 더미화 진행 (type : 3개)

    # 기존 플롯
    plt.figure(figsize=(12, 6))
    plt.plot(X_train['year'], y_train, color='black', label='Train Data')
    plt.xlabel("Year")
    plt.ylabel("Container")


    kor = [i for i in dx if 'KOR' in i]

    model = LinearRegression()
    
    # st.write(X_train[total_var].loc[~X_train['year'].isin(selected_year)])
    # st.write(y_train[[i-2010 for i in range(2010,2024) if i not in selected_year]])
    # X_train.loc[~X_train['year'].isin(del_year)]
    model.fit(X_train[total_var].loc[~X_train['year'].isin(selected_year)], y_train[[i-2010 for i in range(2010,2024) if i not in selected_year]])

    # 예측
    y_pred = model.predict(X_test[total_var])
    plt.plot(
        X_test['year'], y_pred,
        label=f"selected",
        linestyle='-', 
        marker=['o', 's', '*', '^'][k],  # 마커 스타일
        markersize=5
    )
    plt.plot([2024,2025,2026,2027], ARIMA,    label=f"bpa_arima",    linestyle='-',linewidth=3)
    plt.plot([2024,2025,2026,2027], ARIMA_X,    label=f"bpa_arima_x",    linestyle='-', linewidth=3)
    # 모델 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # print("Mean Squared Error (MSE):", mse)
    # print("R-squared (R2):", r2)
    print()

    # 그래프 설정
    plt.xticks(data['year'])
    plt.grid()
    plt.ylim([1e7, 4.5e7])
    plt.legend()
    plt.tight_layout()
    st.write("### 선택 변수 예측 결과 ")
    st.pyplot(plt)
    y_pred_df = pd.DataFrame([round(i) for i in y_pred]).T
    y_pred_df.columns=[2024,2025,2026,2027,2028,2029]
    # st.table(y_pred_df)
    st.write("컨테이너 물동량 예측치 - 단위 TEU")
    st.write(y_pred_df)
    st.write("---")
    # st.write("선택된 WEO 변수:", selected_var)
    # st.write("선택된 국가:", selected_nat)
    # st.write("선택된 연도:", selected_year)
    # st.write(total_var)




# 상태 초기화: 추가된 셀의 수를 저장
if "cell_count" not in st.session_state:
    st.session_state.cell_count = 0

# 사용자가 정의한 함수 (예제)
def my_function(cell_id):
    st.write(f"---")
    st.write(f"#### {cell_id}번째 추가된 회귀분석")
    regression(cell_id)
    # user_input = st.text_input(f"셀 {cell_id}에서 입력:", key=f"input_{cell_id}")
    # st.write(f"입력 값: {user_input}")

# 버튼 클릭 시 새로운 셀 추가
st.write("---")
st.write("")
st.write("### 선택 회귀분석 도구")
st.write("- 셀 추가를 누르면 (제외년도 / 국가 / WEO 변수)를 선택할 수 있는 선택창이 생깁니다. ")
st.write("- 버튼이나 선택창을 눌러서 화면이 흐려져도 선택이 반영됩니다. 기다리지 말고 눌러주세요")
st.write("- 처음 셀 추가를 누르고 변수를 선택하기까지 20~30초 정도 소요됩니다.")
st.write("- 키보드에서 'r' 을 누르면 코드가 다시 실행됩니다.")
if st.button("셀 추가"):
    st.session_state.cell_count += 1

# 현재까지 추가된 셀 출력
for cell_id in range(1, st.session_state.cell_count + 1):
    my_function(cell_id)
