import pandas as pd
import streamlit as st

from MLUtils import *

st.title("Dr. Z.C.M.")
st.title('AI predicts the prognosis of elderly patients in pre-hospital and emergency room')  # 算法名称 and XXX

COL_INPUT = [
    'age', 'gender', 'RR', 'SpO2', 'inhaled O2 ',
    'Temp', 'sBP', 'HR', 'consciousness', 'pNEWS',
    'pMEWS', 'admitted to award ', 'admitted to ICU ', # 'death in 28days '
]

vars = []

btn_predict = None

xgbc = None


# 配置选择变量（添加生成新数据并预测的功能）
def setup_selectors():
    global vars, btn_predict

    if COL_INPUT is not None and len(COL_INPUT) > 0:
        col_num = 3
        cols = st.columns(col_num)

        for i, c in enumerate(COL_INPUT):
            with cols[i % col_num]:
                num_input = st.number_input(f"Please input {c}", value=0, format="%d", key=c)
                vars.append(num_input)

        with cols[0]:
            btn_predict = st.button("Do Predict")

    if btn_predict:
        do_predict()


# 对上传的文件进行处理和展示
def do_processing():
    global xgbc
    pocd = read_csv('./pocd_xgb_3.csv')

    st.text("Dataset Description")
    st.write(pocd.describe())
    if st.checkbox('Show detail of this dataset'):
        st.write(pocd)

    # 分割数据
    X_train, X_test, y_train, y_test = do_split_data(pocd, random_state=18)
    X_train, X_test, y_train, y_test = do_xy_preprocessing(X_train, X_test, y_train, y_test)

    col1, col2 = st.columns(2)

    # 准备模型
    xgbc = XGBClassifier(learning_rate=0.12, n_estimators=12, max_depth=2, min_child_weight=1, gamma=0.1,
                         scale_pos_weight=1)

    # 模型训练、显示结果
    with st.spinner("Training, please wait..."):
        xgbc_result = model_fit_score(xgbc, X_train, y_train)
    with col1:
        st.text("Training Result")
        msg = model_print(xgbc_result, "XGBoost - Train")
        st.write(msg)
        # 训练画图
        fig_train = plt_roc_auc([
            (xgbc_result, 'XGBoost',),
        ], 'Train ROC')
        st.pyplot(fig_train)
    # 模型测试、显示结果
    with st.spinner("Testing, please wait..."):
        xgbc_test_result = model_score(xgbc, X_test, y_test)
    with col2:
        st.text("Testing Result")
        msg = model_print(xgbc_test_result, "XGBoost - Test")
        st.write(msg)
        # 测试画图
        fig_test = plt_roc_auc([
            (xgbc_test_result, 'XGBoost',),
        ], 'Validate ROC')
        st.pyplot(fig_test)


# 对生成的预测数据进行处理
def do_predict():
    global vars
    global xgbc

    # 处理生成的预测数据的输入
    pocd_predict = pd.DataFrame(data=[vars], columns=COL_INPUT)
    pocd_predict = do_base_preprocessing(pocd_predict, with_y=False)
    st.text("Preview for detail of this predict data")
    st.write(pocd_predict)
    pocd_predict = do_predict_preprocessing(pocd_predict)

    # 进行预测并输出
    # XGBoost
    pr = xgbc.predict(pocd_predict)
    pr = pr.astype(np.int)
    st.markdown(r"$\color{red}{XGBoost}$ $\color{red}{Predict}$ $\color{red}{result}$ $\color{red}{" + str(
        COL_Y[0]) + r"}$ $\color{red}{is}$ $\color{red}{" + str(pr[0]) + "}$")


if __name__ == "__main__":
    do_processing()
    setup_selectors()
