from pandas import read_excel
import numpy as np

# читаем файл, получаем нужные столбцы (мой вариант - 121-130)
df = read_excel("data.xls", "Sheet1", skiprows=1, nrows=78, usecols="DR:EA")
#удаляем строки с пустыми значениями
df = df.dropna()
N = len(df)
p = len(df.columns)
print("Z - матрица данных типа объект-признак размером Nxp")
print(f"N={len(df)}, p={len(df.columns)}")
print(df.to_string())
Z = df.values.tolist()
print()
#for j in range(N):
#	for i in range(p):
#		print(f"{Z[j][i]:.3f}\t\t", end="")
#	print()

# lab1 - нам нужны отсюда некоторые данные
# 1a) средние по столбцам, дисперсии по столбцам
#print("1a) средние по столбцам, дисперсии по столбцам")
means = [0] * p
for j in range(p):
	for i in range(N):
		means[j] += Z[i][j]
for i in range(p):
	means[i] /= N
#print("Средние:")
#print(means)

dispersions = [0] * p
for j in range(p):
	for i in range(N):
		dispersions[j] += (Z[i][j] - means[j]) ** 2
for i in range(p):
	dispersions[i] /= N
#print("Дисперсии:")
#print(dispersions)
#print()

# 1б) стандартизированная матрица
#print("1б) стандартизированная матрица")
X = []
for i in range(N):
	l = []
	for j in range(p):
		l.append(0)
	X.append(l)
for j in range(p):
	for i in range(N):
		X[i][j] = (Z[i][j] - means[j]) / (dispersions[j] ** 0.5)
#print("Стандартизованная матрица:")
#for j in range(N):
	#for i in range(p):
		#print(f"{X[j][i]:.3f}\t", end="")
	#print()
#print()

# 1в) ковариационная матрица
#print("1в) ковариационная матрица")
covar = []
for i in range(p):
	l = []
	for j in range(p):
		l.append(0)
	covar.append(l)
for j in range(p):
	for i in range(p):
		t = 0
		for k in range(N):
			t += (Z[k][i] - means[i]) * (Z[k][j] - means[j])
		covar[i][j] = t / N
#print("Ковариационная матрица:")
#for j in range(p):
	#for i in range(p):
		#print(f"{covar[j][i]:.3f}\t", end="")
	#print()
#print()

# 1г) корелляционная матрица
#print("1г) корелляционная матрица")
corel = []
for i in range(p):
	l = []
	for j in range(p):
		l.append(0)
	corel.append(l)
for j in range(p):
	for i in range(p):
		t = 0
		for k in range(N):
			t += X[k][i]*X[k][j]
		corel[i][j] = t / N
print("Корелляционная матрица:")
for j in range(p):
	for i in range(p):
		print(f"{corel[j][i]:.3f}\t", end="")
	print()
print()

# 2) Проверить гипотезу о значимости коэффициентов корреляции между столбцами матрицы данных.
#print("2) Проверить гипотезу о значимости коэффициентов корреляции между столбцами матрицы данных.")
T = []
for i in range(p):
	l = []
	for j in range(p):
		l.append(0)
	T.append(l)
for j in range(p):
	for i in range(p):
		try:
			T[i][j] = (corel[i][j] * (N-2)**0.5) / ((1-corel[i][j]**2)**0.5)
		except ZeroDivisionError:
			T[i][j] = 1e18
#print("Статистика T для каждого коэффицента корелляции:")
#for j in range(p):
	#for i in range(p):
		#if i != j:
			#print(f"{T[j][i]:.3f}\t", end="")
		#else:
			#print("#####\t", end="")
	#print()
#print()

alpha = 0.05
f = N-2 # 57
#print(f"alpha={alpha}, f={f}")
t_table = 2.0024655
#print(f"t_table={t_table}")
#print()

print("Решение о гипотезах")
for j in range(p):
	for i in range(p):
		if i != j:
			if abs(T[i][j]) >= t_table:
				print("H1\t", end="")
			else:
				print("H0\t", end="")
		else:
			print("##\t", end="")
	print()
print()

# lab2
# 1) В качестве y - один из столбцов Z, остальные - в X (коэффицент корелляции между y >= 0.3, H1). Берём 7 столбец за y, тогда X - столбцы 3, 5, 4, 7, 9. 
# 2) Рассмотреть уравнение регрессии со свободным членом
# 3) Найти МНК-оценку
y_index = 6 #7-1
X_indexes = [3,5,7,4,9]

y_list = []
for i in range(N):
    y_list.append(Z[i][y_index])
y = np.array(y_list)
X_list = []
for i in range(N):
	t = []
	for j in X_indexes:
		t.append(Z[i][j])
	X_list.append(t)
X = np.matrix(X_list)
print("Матрица X:")
for j in range(N):
    for i in range(len(X_indexes)):
         print(f"{X_list[j][i]:.3f}\t\t", end="")
    print()
print()
print("Вектор y:")
for elem in y:
	print(f"{elem:.3f}")
print()

def MNKFind(X, X_list, y, y_list):
	X_t = X.transpose()
	b = np.dot(X_t, X)
	a = np.dot(np.dot(np.linalg.inv(np.dot(X_t, X)), X_t), y)
	a_list = a.tolist()[0]
	a = np.array(a_list)
	print("Оценка a:")
	for elem in a_list:
		print(f"{elem:.3f}")
	_y = np.dot(X, a)
	_y_list = _y.tolist()[0]
	print("Вектор расчетных значений зависимой переменной _y:")
	for elem in _y_list:
		print(f"{elem:.3f}")
	e = y - _y
	e_list = e.tolist()[0]
	print("Вектор оценочных отклонений e:")
	for elem in e_list:
		print(f"{elem:.3f}")
	print()
	y_avg = np.mean(y)
	_y_avg = np.mean(_y)
	e_avg = np.mean(e)
	e_sq, y_var = 0, 0
	for i in range(len(X_indexes)):
		e_sq += e_list[i] ** 2
		y_var += (y_list[i] - y_avg) ** 2
	r = 1 - e_sq / y_var
	print("y_avg = ", y_avg)
	print("_y_avg = ", _y_avg)
	print("e_avg = ", e_avg)
	print("Коэффицент детерминации - ", r)

MNKFind(X, X_list, y, y_list)

print("==========================================")
# 4 - тестовый пример
print("Введите матрицу X (5x2):")
X_list = []
y_list = []
for i in range(5):
	X_list.append(list(map(float, input().split())))
print("Введите вектор y длины 5:")
y_list = list(map(float, input().split()))
print(X_list)
print(y_list)
X = np.matrix(X_list)
y = np.array(y_list)
# 5 - проверка тестового примера
MNKFind(X, X_list, y, y_list)
