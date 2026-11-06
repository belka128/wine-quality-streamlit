
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics

path = Path(__file__).parent
@st.cache_resource
def load_model():
    return joblib.load(path / 'model.joblib')
@st.cache_data
def load_data():
    return pd.read_csv(path / 'test.csv')
@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

model = load_model()
data = load_data()
config = json.loads((path / 'config.json').read_text())
features = data.drop(columns='target')
st.title('Wine Quality Prediction')
threshold = st.sidebar.slider('Порог', 0.0, 1.0, float(config['threshold']), 0.01)
tab1, tab2, tab3, tab4 = st.tabs([
    'Оценка вина', 'Модельные метрики', 'Пакетное предсказание', 'Коэффициенты'
])

with tab1:
    st.header('Оценка одного вина')
    values = {}
    for col in features.columns:
        if col == 'is_red':
            values[col] = int(st.checkbox('Красное вино', value=True))
        else:
            values[col] = st.number_input(
                col,
                min_value=float(features[col].min()),
                max_value=float(features[col].max()),
                value=float(features[col].median())
            )

    wine = pd.DataFrame([values])
    probability = model.predict_proba(wine)[0, 1]
    prediction = int(probability >= threshold)

    st.metric('Вероятность хорошего вина', f'{probability:.2%}')
    if prediction == 1:
        st.success('Вино хорошее')
    else:
        st.warning('Вино не относится к хорошим')
        

with tab2:
    st.header('Модельные метрики')
    X_test = data.drop(columns='target')
    y_test = data['target']
    p = test_proba(model, X_test)
    pred = (p >= threshold).astype(int)

    scores = {
        'Accuracy': metrics.accuracy_score(y_test, pred),
        'Precision': metrics.precision_score(y_test, pred, zero_division=0),
        'Recall': metrics.recall_score(y_test, pred),
        'F1': metrics.f1_score(y_test, pred),
        'ROC-AUC': metrics.roc_auc_score(y_test, p),
        'PR-AUC': metrics.average_precision_score(y_test, p)
    }
    st.dataframe(pd.DataFrame([scores]).round(4))
    st.write('При увеличении порога модель рекомендует меньше вин, но Precision обычно растёт. При уменьшении порога модель находит больше хороших вин, но чаще ошибается.')

    cm = metrics.confusion_matrix(y_test, pred)
    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    fpr, tpr, _ = metrics.roc_curve(y_test, p)
    ax.plot(fpr, tpr, label='ROC')
    fpr_point = ((y_test == 0) & (pred == 1)).sum() / (y_test == 0).sum()
    tpr_point = metrics.recall_score(y_test, pred, zero_division=0)
    ax.scatter(fpr_point, tpr_point, color='red', label='Current threshold', zorder=3)
    ax.plot([0, 1], [0, 1], '--', label='Random')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    precision, recall, _ = metrics.precision_recall_curve(y_test, p)
    ax.plot(recall, precision, label='PR')
    pr_point = metrics.precision_score(y_test, pred, zero_division=0)
    rec_point = metrics.recall_score(y_test, pred, zero_division=0)
    ax.scatter(rec_point, pr_point, color='red', label='Current threshold', zorder=3)
    ax.axhline(y_test.mean(), linestyle='--', label='Random')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


with tab3:
    st.header('Пакетное предсказание')
    uploaded = st.file_uploader('Загрузите CSV-файл', type='csv')
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        cols = list(features.columns)
        missing = [col for col in cols if col not in df.columns]
        if missing:
            st.error(f'Отсутствуют признаки: {", ".join(missing)}')
        else:
            X = df[cols]
            p = model.predict_proba(X)[:, 1]
            result = df.copy()
            result['probability'] = p
            result['prediction'] = (p >= threshold).astype(int)

            st.dataframe(result)
            st.download_button(
                'Скачать предсказания',
                result.to_csv(index=False).encode('utf-8'),
                file_name='predictions.csv',
                mime='text/csv'
            )


with tab4:
    st.header('Коэффициенты модели')
    lr = model.named_steps['logisticregression']
    coefs = pd.DataFrame({
        'Feature': features.columns,
        'Coefficient': lr.coef_[0]
    })
    coefs = coefs.sort_values('Coefficient')
    st.dataframe(coefs)
    st.write(f'Сильнее всего за хорошее вино: {coefs.iloc[-1]["Feature"]}')
    st.write(f'Сильнее всего против: {coefs.iloc[0]["Feature"]}')
    st.write('Коэффициенты относятся к масштабированным признакам.')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(coefs['Feature'], coefs['Coefficient'])
    ax.set_xlabel('Coefficient')
    st.pyplot(fig)
    plt.close(fig)

