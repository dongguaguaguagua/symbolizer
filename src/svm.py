from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from common import load_data


if __name__ == "__main__":
    print("loading data...")
    data, labels = load_data("./data/HASYv2")

    print("processing data...")
    # 只保留第一个颜色通道
    data = data[:, :, 0, :]
    # 将每个图像展平为一维向量
    num_samples = data.shape[2]
    flattened_data = data.reshape(num_samples, -1)
    print("sample numbers:", num_samples)

    print("split dataset...")
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        flattened_data, labels, test_size=0.2, random_state=42
    )

    # 标准化数据
    print("standarlize data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练 SVM 模型
    print("training SVM model...")
    svm_model = SVC(kernel="rbf", gamma="scale", C=1, verbose=True)
    svm_model.fit(X_train, y_train)

    # 评估模型
    print("evaluating model...")
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    print(data.shape)  # (32, 32, 168233)
    print(labels.shape)  # (168233, 1)
