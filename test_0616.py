import streamlit as st
import joblib
import numpy as np

# 加载模型
model = joblib.load('best_models_reduced.pkl')

# 页面配置
st.set_page_config(page_title="糖尿病风险预测", page_icon="", layout="centered")
st.title("糖尿病风险预测")
st.caption("⚠️ 免责声明：模型仅用于课堂演示，预测结果不构成任何医疗诊断或健康建议。数据不会被保存。")

# 年龄映射字典
age_map = {
    "18-24岁": 1,
    "25-29岁": 2,
    "30-34岁": 3,
    "35-39岁": 4,
    "40-44岁": 5,
    "45-49岁": 6,
    "50-54岁": 7,
    "55-59岁": 8,
    "60-64岁": 9,
    "65-69岁": 10,
    "70-74岁": 11,
    "75-79岁": 12,
    "80岁及以上": 13
}

# 收入映射字典（包含默认值）
income_map = {
    "不清楚（使用默认值：10-13.5万）": 3,  # 默认值，可根据实际数据调整
    "少于7万人民币": 1,
    "7-10万人民币": 2,
    "10-13.5万人民币": 3,
    "13.5-17万人民币": 4,
    "17-23.5万人民币": 5,
    "23.5-34万人民币": 6,
    "34-50万人民币": 7,
    "50万人民币以上": 8
}

# 使用表单
with st.form("prediction_form"):
    st.subheader("请输入您的信息")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.selectbox("年龄段", list(age_map.keys()), index=0)  # 默认18-24岁
        sex = st.radio("性别", ["女", "男"], index=0, horizontal=True)
        bmi = st.number_input("BMI 指数", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
        high_bp = st.radio("是否高血压", ["否", "是"], index=0, horizontal=True)
    
    with col2:
        gen_hlth = st.selectbox("整体健康自评", ["很好", "较好", "一般", "较差", "很差"], index=0)
        income = st.selectbox("全家年收入", list(income_map.keys()), index=0)  # 默认"不清楚"
        chol_check = st.radio("5年内是否做过胆固醇检查", ["否", "是"], index=0, horizontal=True)
    
    submitted = st.form_submit_button("预测我的风险", use_container_width=True, type="primary")

if submitted:
    # 数据编码
    gen_hlth_map = {"很好": 1, "较好": 2, "一般": 3, "较差": 4, "很差": 5}
    
    input_data = np.array([[
        gen_hlth_map[gen_hlth],
        1 if high_bp == "是" else 0,
        bmi,
        age_map[age],
        income_map[income],
        1 if sex == "男" else 0,
        1 if chol_check == "是" else 0
    ]])
    
    # 预测
    probability = model.predict_proba(input_data)[0][1]
    prediction = "是" if probability > 0.5 else "否"
    
    # 显示结果
    st.divider()
    st.subheader("预测结果")
    
    # 风险等级划分
    if probability < 0.3:
        risk_level = "极低风险"
        color = "success"
        emoji = "🟢"
        advice = "你的风险非常低！继续保持健康的生活方式吧～"
    elif probability < 0.5:
        risk_level = "低风险"
        color = "success"
        emoji = "💚"
        advice = "风险较低，但也要注意保持规律作息和健康饮食哦！"
    elif probability < 0.7:
        risk_level = "高风险"
        color = "orange"
        emoji = ""
        advice = "风险较高，建议咨询医生，注意监测血糖水平。"
    else:
        risk_level = "极高风险"
        color = "error"
        emoji = "🔴"
        advice = "风险很高，强烈建议尽快就医检查！"
    
    # 显示主要结果
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(label="糖尿病患病概率", value=f"{probability:.2%}")
        
        # 进度条
        st.progress(probability)
        
        # 风险等级
        if probability < 0.5:
            st.success(f"{emoji} **{risk_level}** | 预测结果：**{prediction}**")
        else:
            st.error(f"{emoji} **{risk_level}** | 预测结果：**{prediction}**")
        
        # 个性化建议
        st.info(f"💡 {advice}")
    
    # 显示输入摘要
    with st.expander("📋 查看输入信息摘要"):
        st.write(f"""
        - **年龄段**: {age}
        - **性别**: {sex}
        - **BMI**: {bmi}
        - **整体健康**: {gen_hlth}
        - **高血压**: {high_bp}
        - **收入水平**: {income}
        - **胆固醇检查**: {chol_check}
        """)
