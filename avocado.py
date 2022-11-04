
from matplotlib.pyplot import subplot
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from math import sqrt
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pmdarima import auto_arima
from sklearn.ensemble import ExtraTreesRegressor
import pickle
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_components
import streamlit as st


def convert_month(month):
    if month == 3 or month == 4 or month == 5:
        return 0
    elif month == 6 or month == 7 or month == 8:
        return 1
    elif month == 9 or month == 10 or month == 11:
        return 2
    else:
        return 3

def replace_outlier(df, col):
    Q1 = np.quantile(df[col].dropna(),0.25)
    Q3 = np.quantile(df[col].dropna(),0.75)
    iqr = Q3 - Q1
    df.loc[(df[col] > (Q3 + 1.5*iqr)) | (df[col] < (Q1 - 1.5*iqr)), col] = df[col].median()

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return mse, rmse, mae

def dummies(x,df):
    temp = pd.get_dummies(df[x])
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

def procesing_reg(df):  
    # Create more date, month columns for analysis
    df['Date'] = pd.to_datetime(df['Date'])
    df['day']=df['Date'].dt.day
    df['month']=df['Date'].dt.month
    df['Season'] = df['month'].apply(lambda x: convert_month(x))

    df_new = df[['Total Volume', 'Total Bags', 'type', 'year', 'region', 'day', 'month','Season', 'AveragePrice']]
    # Drop na
    df_new.dropna(axis = 1, how ='all', inplace = True)
    # dropping duplicate values
    df_new.drop_duplicates(keep=False, inplace=True)
    df_new = df.copy().reset_index(drop=True)

    X_col = ['Total Volume', 'Total Bags', 'type', 'year', 'region', 'day', 'month','Season']
    X = df_new[X_col]
    y = df_new['AveragePrice']

    # replace outlier to median
    columns = ['Total Volume', 'Total Bags']
    for col in columns:
        replace_outlier(X, col)

    # categorical data type conversion
    lst_categories = ['type','year', 'region', 'day', 'month','Season']
    for col in lst_categories:
        X[col] = pd.Categorical(X[col])

    # Label Encoder for 'type'
    encoder = LabelEncoder()
    X['type'] = encoder.fit_transform(X['type'])

    # convert categorical attribute to numeric type: get_dummies()
    X = dummies('region',X)

    # Scaler
    scaler = StandardScaler()
    X_arr = scaler.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=X.columns)
    
    return X, y

def convert_df(df):
    return df.to_csv().encode('utf-8')

def selec_region(df, region, types):
    df['Date'] = df['Date'].apply(lambda x: x[:7])
    df = df[(df.type == types) & (df.region == region)]
    df = df[['Date', 'AveragePrice']]

    return df


# Choose ToTalVolume bc it has high corr with '4046','4225','4770','Small Bags','Large Bags','XLarge Bags'
df = pd.read_csv('data/avocado.csv', index_col=0)

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')

# -----------Part 1: Regression----------------
# Create more date, month columns for analysis
df_reg = df.copy()
df_reg['Date'] = pd.to_datetime(df_reg['Date'])
df_reg['day']=df_reg['Date'].dt.day
df_reg['month']=df_reg['Date'].dt.month
df_reg['Season'] = df_reg['month'].apply(lambda x: convert_month(x))

# Create new dataframe with best features
df_new = df_reg[['Total Volume', 'Total Bags', 'type', 'year', 'region', 'day', 'month','Season', 'AveragePrice']]
df_new = df_new.reset_index(drop=True)

X_col = ['Total Volume', 'Total Bags', 'type', 'year', 'region', 'day', 'month','Season']
X = df_new[X_col]
y = df_new['AveragePrice']

# categorical data type conversion
lst_categories = ['type', 'region', 'month', 'Season']
for col in lst_categories:
    X[col] = pd.Categorical(X[col])

# Label Encoder for 'type'
encoder = LabelEncoder()
X['type'] = encoder.fit_transform(X['type'])

def dummies(x,df):
    temp = pd.get_dummies(df[x])
    df = pd.concat([df, temp], axis = 1)
    df.drop([x], axis = 1, inplace = True)
    return df

# convert categorical attribute to numeric type: get_dummies()
X = dummies('region',X)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_arr = scaler.fit_transform(X)
X = pd.DataFrame(X_arr, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.3)
# create new model with adjustable parameters
reg_model = ExtraTreesRegressor(n_estimators=120, random_state=0).fit(X_train, y_train)

import pickle
# Save model
pkl_filename = 'data/reg_model_avocado.pkl'
with open(pkl_filename, 'wb') as file:  
    pickle.dump(reg_model, file)

y_pred = reg_model.predict(X_test)
r2 = r2_score(y_pred, y_test)
mae = mean_squared_error(y_pred, y_test)

metrics = pd.DataFrame([{
    'Model': 'ExtraTreesRegressor',
    'r2_score': r2,
    'MAE': mae
}])
metrics
metrics.to_csv('data/metrics_res.txt')

# Load model regression
pkl_filename = 'data/reg_model_avocado.pkl'
with open(pkl_filename, 'rb') as file:  
    reg_model = pickle.load(file)

# Evaluation
metrics_res = pd.read_table('data/metrics_res.txt', sep=',', index_col=0)

# -----------Part 2: Facebook Prophec----------------
# Filter Avocado - California
# Make new dataframe from original dataframe: data
df_organic = selec_region(df, 'California', 'organic')
df_conventional = selec_region(df, 'California', 'conventional')
df_losangeles = selec_region(df, 'LosAngeles', 'organic')
# Group theo month
agg = {'AveragePrice': 'mean'}
df_organic = df_organic.groupby(df_organic['Date']).aggregate(agg).reset_index()
df_organic.columns = ['ds', 'y']
train_organic = df_organic.drop(df_organic.index[-10:])
test_organic = df_organic.drop(df_organic.index[0:-10])
df_conventional = df_conventional.groupby(df_conventional['Date']).aggregate(agg).reset_index()
df_conventional.set_index('Date', inplace=True)
train_conentional = df_conventional.loc['2015-01-01':'2017-06-01']
test_conentional = df_conventional.loc['2017-06-01':]
df_losangeles = df_losangeles.groupby(df_losangeles['Date']).aggregate(agg).reset_index()
df_losangeles.columns = ['ds', 'y']

# Load model
## Model prophec
pkl_filename = 'data/avocado_organic_prophec.pkl'
with open(pkl_filename, 'rb') as file:  
    model_prohet = pickle.load(file)
## Model ARIMA
pkl_filename = 'data/avocado_conventional_arima.pkl'
with open(pkl_filename, 'rb') as file:  
    model_arima = pickle.load(file)
## Model prophec for Losangeles
pkl_filename = 'data/avocado_organic_los_prophec.pkl'
with open(pkl_filename, 'rb') as file:  
    model_prophec_los = pickle.load(file)
## Model prophec for Losangeles next 5 year
pkl_filename = 'data/avocado_organic_los_prophec_5year.pkl'
with open(pkl_filename, 'rb') as file:  
    model_prophec_los_5year = pickle.load(file)
# Evaluation
metrics_Prophet = pd.read_table('data/avocado_organic_prophec.txt', sep=',', index_col=0)
metrics_arima = pd.read_table('data/avocado_conventional_arima.txt', sep=',', index_col=0)
metrics_prophec_los = pd.read_table('data/avocado_organic_los_prophec.txt', sep=',', index_col=0)
# Create 12 next month to predict
months = pd.date_range('2017-06-01', '2019-03-01',
                      freq='MS').strftime('%Y-%m-%d').to_list()
future_12 = pd.DataFrame(months)
future_12.columns = ['ds']
future_12['ds'] = pd.to_datetime(future_12['ds'])
# make a forecast
forecast_prophect = model_prohet.predict(future_12)
y_pred_prophec = forecast_prophect['yhat'].values[:10]
y_test_prophec = test_organic['y'].values
y_test_arima = test_conentional
y_pred_arima = model_arima.predict(n_periods=len(test_conentional))
y_pred_arima = pd.DataFrame(y_pred_arima, index = y_test_arima.index, columns=['Prediction'])

# make predict other regions

# GUI
st.title("ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE")
st.caption(' Đồ án Tốt nghiệp Data Science tại Trung Tâm Tin Học - Trường Đại học Khoa học Tự nhiên TP.HCM')
st.header("Avocado price prediction")

menu = ["Introduce", "Data Understanding", "Build Project", "New Prediction"]
choice = st.sidebar.selectbox('Content', menu)

if choice == 'Introduce':
    st.subheader('Business Objective')
    st.write("""**Vấn đề hiền tại:** Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn *(Small/Large/XLarge Bags)*, và có 3 PLU (Product Look Up) khác nhau *(4046, 4225, 4770)*. Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng""")
    st.write("""
    **Mục tiêu/ Vấn đề:**
    => Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ => xem xét việc mở rộng sản xuất, kinh doanh.
    """)
    st.image('data/avocado.jpg')
elif choice == 'Data Understanding':
    st.subheader('Data Understanding')
    st.write('About the Dataset:')
    st.write("""
    - Dữ liệu được lấy trực tiếp từ máy tính tiền của các nhà bán lẻ dựa trên doanh số bán lẻ thực tế của bơ Hass.
    - Dữ liệu đại diện cho dữ liệu lấy từ máy quét bán lẻ hàng tuần cho lượng bán lẻ (National retail volume- units) và giá bơ từ tháng 4/2015 đến tháng 3/2018.
    - Giá Trung bình (Average Price) trong bảng phản ánh giá trên một đơn vị (mỗi quả bơ), ngay cả khi nhiều đơn vị (bơ) được bán trong bao.
    - Mã tra cứu sản phẩm - Product Lookup codes (PLU’s) trong bảng chỉ dành cho bơ Hass, không dành cho các sản phẩm khác.
    - Toàn bộ dữ liệu được đổ ra và lưu trữ trong tập tin avocado.csv với 18249 record. Với các cột:
    1. Date - ngày ghi nhận
    2. AveragePrice – giá trung bình của một quả bơ
    3. Type - conventional / organic – loại: thông thường/ hữu cơ
    4. Region – vùng được bán
    5. Total Volume – tổng số bơ đã bán
    6. 4046 – tổng số bơ có mã PLU 4046 đã bán
    7. 4225 - tổng số bơ có mã PLU 4225 đã bán
    8. 4770 - tổng số bơ có mã PLU 4770 đã bán
    9. Total Bags – tổng số túi đã bán
    10. Small/Large/XLarge Bags – tổng số túi đã bán theo size
    11. Có hai loại bơ trong tập dữ liệu và một số vùng khác nhau. Điều này cho phép chúng ta thực hiện tất cả các loại phân tích cho các vùng khác nhau, hoặc phân tích toàn bộ nước Mỹ theo một trong hai loại bơ.
    """)

elif choice == 'Build Project':
    st.subheader('Build model')

    with st.sidebar:
        choose_model = st.radio(
            "Choose one of the models",
            ("Regresion", "Time series")
        )
    if choose_model == 'Regresion':
        #------ Regresion Model----------
        st.write("#### Regresion Model")
        st.write('Show data')
        st.dataframe(df.head())
        st.dataframe(df.tail())
        # data visualization
        st.write('#### visualization')
        ## Average avocado price of each variety by year
        price_type = df.groupby(['type', 'year'])['AveragePrice'].mean().reset_index()
        fig, ax = plt.subplots()
        fig.suptitle('Average avocado price of each variety by year')
        sns.barplot(x='year', y='AveragePrice', data=price_type, hue='type')
        st.pyplot(fig)
        ## Analysis of Average Prices
        byDate=df.groupby('Date').mean()
        st.line_chart(byDate['AveragePrice'], use_container_width=True)
        # evaluation
        st.write('#### Evaluation')
        st.table(metrics_res)
        st.write('R-square = 0.9099 and low MAE (0.0131), the model is suitable for prediction')
        st.image('data/actual_prediction_regression.png')
    else:
        st.write("#### Time series")
        st.caption('Building an avocado price prediction model for the California region')
        tab1, tab2 = st.tabs(["📈 Conventional", "🗃 Organic"])
        #----------conventional---------
        tab1.write('Some data')
        tab1.dataframe(df_conventional.head())

        # Evaluation
        tab1.write('Evaluation')
        tab1.table(metrics_arima)
        tab1.write(test_conentional['AveragePrice'].describe())
        tab1.write('mean test = 1.2964, std = 0.2594, both MSE and MAE are smaller than std => ok')
        # Visulaize the result
        tab1.write("##### Visualization")
        fig, ax = plt.subplots()    
        ax.plot(y_test_arima, label='AveragePrice')
        ax.plot(y_pred_arima, label='AveragePrice Prediction')
        ax.legend()  
        ax.set_title('AveragePrice vs AveragePrice Prediction', fontsize=12)  
        tab1.pyplot(fig)

        #----------organic---------
        tab2.write('Some data')
        tab2.dataframe(df_organic.head())

        # Evaluation
        tab2.write('Evaluation')
        tab2.table(metrics_Prophet)
        tab2.write('Describe')
        tab2.write(test_organic['y'].describe())
        tab2.write('mean test = 1.9163, std = 0.2098, both MSE and MAE are smaller than std => ok')
        # Visulaize the result
        tab2.write("##### Visualization: AveragePrice vs AveragePrice Prediction")

        fig, ax = plt.subplots()    
        ax.plot(y_test_prophec, label='AveragePrice')
        ax.plot(y_pred_prophec, label='AveragePrice Prediction')
        ax.legend()    
        tab2.pyplot(fig)
 

elif choice == 'New Prediction':
    st.subheader('New Prediction')
    with st.sidebar:
        model_predict = st.radio(
            "Choose one of the models",
            ("Regresion", "Time series")
        )
    # model_predict = st.radio("Which model do you choose to predict?",('Regresion', 'Time series'))

    if model_predict == 'Regresion':
        st.success('You selected Regresion!')
        st.subheader('Select data')
        type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
        if type=="Upload": 
            uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
            if uploaded_file is not None:
                test = pd.read_csv(uploaded_file)
                st.write('Show data')
                st.dataframe(test.head())
                data_test = procesing_reg(test)
                X_new = data_test[0]
                y_new = data_test[1]
                y_pred_new = reg_model.predict(X_new)

                st.code("predicted results: " + str(y_pred_new))

                # Tạo dataframe mới chứa y_pred
                y_pred_new = pd.DataFrame(y_pred_new, columns=['prediction'])
                # data = pd.concat([test, y_pred_new], axis=1)

                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(y_pred_new)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='data.csv',
                    mime='text/csv',)
        
        
        if type=="Input":        
            total_volume = st.number_input('Total Volume',value=64236.62, key=1)
            total_bags = st.number_input('Total Bags', value=8696.87, key=2)
            types = st.radio("Type",('conventional', 'organic'))
            date = st.date_input("Date",datetime.date(2015, 12, 27))
            day = date.day
            month = date.month
            year = date.year
            season = convert_month(month)
            region = st.selectbox("Region",options=df['region'].unique())
            

            if st.button('Predict'):
                types_avocado = 0 if types == 'conventional' else 1
                # make new prediction
                new_data = pd.DataFrame([{'Total Volume': total_volume,
                    'Total Bags': total_bags,
                    'type': types_avocado,
                    'year': year,
                    'day': day,
                    'month': month,
                    'Season': season}])

                
                # Show result
                st.success('Success!', icon="✅")
                st.write('Input:')
                new_data.to_csv('new_data.csv')               
                
                # Create more region columns
                regions_arr = df['region'].unique()
                for i in regions_arr:
                    if i == region:
                        new_data[i] = 1
                    else:
                        new_data[i] = 0
                    
                st.dataframe(new_data)
                # # Scale data
                # scaler = StandardScaler()
                # new_data = scaler.fit_transform(new_data)
                results = reg_model.predict(new_data)
                st.code('predicted results: ' + str(results))

    
    else:
        st.success('You selected Time series!')

        region_selec = st.radio('Select the region to predict California or another region', ('California', 'LosAngeles'), horizontal = True)
        if region_selec == 'California':
            st.write('#### Make new prediction for the future in California')
            next_time = st.radio('How long do you want to predict?', ('12 tháng', '5 năm'))
            type_avocado = st.radio('Type', ('organic', 'conventional'))
            
            if st.button('Predict'):

                st.balloons()
                if type_avocado == 'conventional':
                    
                    if next_time == '12 tháng':
                        next_time = 12
                    else:
                        next_time = 360

                    future_price_nex_time = model_arima.predict(n_periods=len(test_organic)+ next_time)
                    st.code(future_price_nex_time)
                    
                    # # download file
                    future_price_nex_time_df = pd.DataFrame(future_price_nex_time, columns=['predict_AveragePrice'])
                    csv = convert_df(future_price_nex_time_df)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='future_price_next_time_' + str(next_time) + '.csv',
                        mime='text/csv',)
            
                    
                    # visualzation
                    fig, ax = plt.subplots()
                    ax.plot(future_price_nex_time, label='Prediction')
                    ax.set_title('Prediction for the next' + str(next_time) + 'months')
                    # ax.xticks(rotation='vertical')
                    ax.legend()
                    st.pyplot(fig)
                
                else:
                    # predict with facebook prophec
                    if next_time == '12 tháng':
                        st.write('##### Next 12 months')
                        # Create 12 next month to predict
                        months = pd.date_range('2017-06-01', '2019-03-01',
                                            freq='MS').strftime('%Y-%m-%d').to_list()
                        future_12 = pd.DataFrame(months)
                        future_12.columns = ['ds']
                        future_12['ds'] = pd.to_datetime(future_12['ds'])
                        
                        forecast = model_prohet.predict(future_12)


                        # visualzation
                        figure = model_prohet.plot(forecast, xlabel='Date', ylabel='Price')
                        a = add_changepoints_to_plot(figure.gca(), model_prohet, forecast)
                        st.pyplot(figure)

                        # expected trend in the future
                        figure2 = model_prohet.plot_components(forecast)
                        st.pyplot(figure2)
                        
                    
                    else:
                        st.write('##### Long-term prediction for the next 5 years => Consider whether to expand cultivation/production, and trading')
                        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False) 
                        m.fit(df_organic)
                        future = m.make_future_dataframe(periods=60, freq='M')
                        forecast = m.predict(future) 

                        fig = m.plot(forecast) 
                        a = add_changepoints_to_plot(fig.gca(), m, forecast)
                        st.pyplot(fig)

                        fig1 = m.plot_components(forecast)
                        st.pyplot(fig1)

                        fig2, ax = plt.subplots()
                        ax.plot(df_organic['y'], label='True values')
                        ax.plot(forecast['yhat'], label='Price with next 60 months prediction', 
                                color='red')
                        ax.legend()
                        st.pyplot(fig2)
        else:
            st.write('#### Make new prediction for the future in California')

            st.write('Some data')
            st.dataframe(df_organic.head())

            # Evaluation
            st.write('Evaluation')
            st.table(metrics_prophec_los)
            # Visulaize the result
            st.write("##### Visualization: AveragePrice vs AveragePrice Prediction")

            forecast_prophect = model_prophec_los.predict(future_12)
            y_pred_prophec = forecast_prophect['yhat'].values[:10]
            y_test_prophec = test_organic['y'].values
            fig, ax = plt.subplots()    
            ax.plot(y_test_prophec, label='AveragePrice')
            ax.plot(y_pred_prophec, label='AveragePrice Prediction')
            ax.legend()    
            st.pyplot(fig)

            st.write('##### Long-term prediction for the next 5 years => Consider whether to expand cultivation/production, and trading')
            future = model_prophec_los_5year.make_future_dataframe(periods=60, freq='M')
            forecast = model_prophec_los_5year.predict(future) 

            fig = model_prophec_los_5year.plot(forecast) 
            a = add_changepoints_to_plot(fig.gca(), model_prophec_los_5year, forecast)
            st.pyplot(fig)

            fig1 = model_prophec_los_5year.plot_components(forecast)
            st.pyplot(fig1)

            fig2, ax = plt.subplots()
            ax.plot(df_losangeles['y'], label='True values')
            ax.plot(forecast['yhat'], label='Price with next 60 months prediction', 
                    color='red')
            ax.legend()
            st.pyplot(fig2)
            st.write('Avocado prices tend to increase in the next 5 years => Very convenient to expand cultivation and business')