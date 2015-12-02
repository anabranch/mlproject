from kaggle_helper import KaggleHelper
import loader
kh = KaggleHelper("test.db")

z = loader.XY5(kh)

print(z['X_train'].shape)
