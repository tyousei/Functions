

### 贝叶斯集成函数
data是子模型的预测结果，为$n*i$二维列表，其中n是子模型数量，i为样本数
```
def bayesian(data, thresh, n):
    '''
    data: predicts of sub-models which is a 2-D list
    thresh: 
    n: number of the sub-models
    '''
    P_x_F = list(map(lambda i: np.exp(-thresh[i] / data[i]), range(n)))
    P_x_N = list(map(lambda i: np.exp(-data[i] / thresh[i]), range(n)))

    p_F_x = list(map(lambda i: P_x_F[i] * 0.01 / (P_x_N[i] * (1-0.01) + P_x_F[i] * 0.01), range(n)))

    sum_P_x_F = np.sum(list(map(lambda i: np.exp(P_x_F[i] * 0.5), range(n))), 0)
    BIC = np.sum(list(map(lambda i: p_F_x[i] * np.exp(P_x_F[i]*0.5) / sum_P_x_F, range(n))), 0)
    return BIC
 ```
### 核密度估计
输入样本及置信度，返回该置信度下的阈值
```
import statsmodels.api as sm
from scipy.interpolate import interp1d
def ksdensity_ICDF(x, p):
    '''
    x: samples
    p: confidence
    '''
    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()
    # interpolate KDE CDF to get support values
    fint = interp1d(kde.cdf, kde.support)
    return fint(p)
```

### 故障检测指标计算
返回检出率和误报率
```
def index(label, pred):
    # 计算检出率、误报率
    FDR = np.sum(pred[np.nonzero(label)]) / pred[np.nonzero(label)].shape  # 检出率
    FPR = np.sum(pred[np.nonzero(-label + 1)]) / pred[np.nonzero(-label + 1)].shape  # 误报率
    return FDR, FPR
 ```

### 图片旋转及镜像翻转
```
def rotate(img, angle):
    # 图片旋转操作
    rotate = cv2.getRotationMatrix2D((32, 32, angle, 1)  # 参数：旋转中心点, 旋转角度, 缩放比例
    rotate_img = cv2.warpAffine(img, rotate, (64, 64))
    return rotate_img 
    
def flip(img):
    # 图片镜像操作
    return np.concatenate((img, cv2.flip(img, 1)), 0)

def rotate(img):
    # 图片旋转操作，返回图片90°旋转四次的图片集合
    angle = 0
    output = []
    for i in range(4):
        angle += 90
        rotate = cv2.getRotationMatrix2D((32, 32, angle, 1)  # 参数：旋转中心点, 旋转角度, 缩放比例
        output.append(cv2.warpAffine(img, rotate, (64, 64)))
    return np.array(output)
```


