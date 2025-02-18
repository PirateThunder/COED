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
print("Средние:")
print(means)

dispersions = [0] * p
for j in range(p):
	for i in range(N):
		dispersions[j] += (Z[i][j] - means[j]) ** 2
for i in range(p):
	dispersions[i] /= N
print("Дисперсии:")
print(dispersions)
print()

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
print("Стандартизованная матрица:")
for j in range(N):
	for i in range(p):
		print(f"{X[j][i]:.3f}\t", end="")
	print()
print()

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

d = 0
for i in range(p):
	for j in range(p):
		if i == j:
			continue
		d += corel[i][j] ** 2
d *= len(Z)

print(f"Проверка диагонали: {d:.3f}")

def sgn(n):
	if n >= 0:
		return 1
	else:
		return -1

def yakobi(A):
	eps = 1e-2	

	# шаг 1
	n = len(A)
	t = []
	for i in range(n):
		t_ = []
		for j in range(n):
			t_.append(0)
		t.append(t_)
	for i in range(n):
		t[i][i] = 1
	# шаг 2
	a_0 = 0
	for j in range(1, n): 
		for i in range(j):
			if i == j:
				continue
			a_0 += A[i][j] ** 2
	a_0 = ((2 * a_0) ** 0.5) / n
	a_k = a_0

	while True:
		flag = True
		for j in range(1, n):
			for i in range(j):
				if A[i][j] > abs(eps * a_0):
					flag = False
		if flag:
			break
		# шаг 3
		a_pq = -1
		p, q = -1, -1
		for j in range(1, n):
			for i in range(j):
				if i == j:
					continue
				if abs(A[i][j]) > a_k and abs(A[i][j]) > a_pq:
					a_pq = abs(A[i][j])
					p = i
					q = j
		# шаг 4
		y, x, s, c = 0, 0, 0, 0
		if p != -1:
			y = (A[p][p] - A[q][q]) / 2
			if y == 0:
				x = -1
			else:
				x = -sgn(y) * A[p][q] / (A[p][q]**2 + y**2) ** 0.5 #
			s = x / ((2 * (1 + (1 - x**2) ** 0.5))**0.5) #
			c = (1 - s**2)**0.5

			for i in range(n):
				if i != p and i != q:
					z_1 = A[i][p]
					z_2 = A[i][q]
					A[q][i] = z_1*s + z_2*c
					A[i][q] = A[q][i]
					A[i][p] = z_1*c - z_2*s
					A[p][i] = A[i][p]
			z5 = s**2
			z6 = c**2
			z7 = s*c
			v1 = A[p][p]
			v2 = A[p][q]
			v3 = A[q][q]
			A[p][p] = v1*z6 + v3*z5 - 2*v2*z7
			A[q][q] = v1*z5 + v3*z6 + 2*v2*z7
			A[p][q] = (v1-v3)*z7 + v2*(z6-z5)
			A[q][p] = A[p][q]
			for i in range(n):
				z3 = t[i][p]
				z4 = t[i][q]
				t[i][q] = z3*s + z4*c
				t[i][p] = z3*c - z4*s
		# шаг 5
		a_k /= n**2
	return t


t = yakobi(corel)
lamda = [0] * p
for i in range(p):
	lamda[i] = corel[i][i]

# сортировка
for i in range(len(lamda)):
	for j in range(i+1, len(lamda)):
		if lamda[i] < lamda[j]:
			tmp = lamda[i]
			lamda[i] = lamda[j]
			lamda[j] = tmp
		for q in range(len(t)):
			tmp = lamda[i]
			lamda[i] = lamda[j]
			lamda[j] = tmp

# транспонирование
for i in range(len(t)):
	for j in range(i, len(t)):
		tmp = t[i][j]
		t[i][j] = t[j][i]
		t[j][i] = tmp
print("Собственные числа:")
for x in lamda:
	print(f"{x:.3f}", end="\t")
print()
print("Собственные вектора:")
for i in range(p):
	for j in range(p):
		print(f"{t[i][j]:.3f}", end="\t")
	print()

# поиск y
y = []
for i in range(N):
	t_ = []
	for j in range(p):
		t_.append(0)
	y.append(t_)
for i in range(p):
	yy = [0] * N
	for k in range(p):
		for j in range(N):
			yy[j] += X[j][k] * t[i][k]
	for j in range(N):
		y[j][i] = yy[j]
print("y:")
for i in range(p):
	for j in range(p):
		print(f"{y[i][j]:.3f}", end="\t")
	print()

def average(y):
	N = len(y)
	p = len(y[0])
	avg = [0] * p
	for i in range(p):
		for j in range(N):
			avg[i] += y[j][i]
		avg[i] /= N
	return avg

def compute_variance(y):
	N = len(y)
	p = len(y[0])
	var = [0] * p
	avg = average(y)
	for i in range(p):
		for j in range(N):
			var[i] += (y[j][i] - avg[i])**2
		var[i] /= N
	return var

var_X = compute_variance(X)
var_y = compute_variance(y)
sum_X = 0
for x in var_X:
	sum_X += x
sum_y = 0
for x in var_y:
	sum_y += x
print(f"sum_X={sum_X:.3f}, sum_y={sum_y:.3f}")

def covar(y):
	N = len(y)
	p = len(y[0])
	avg = average(y)
	cor = []
	for i in range(p):
		t_ = []
		for j in range(p):
			t_.append(0)
		cor.append(t_)
	for i in range(p):
		for j in range(p):
			x = 0
			for k in range(N):
				x += (y[k][i] - avg[i])**2
			x /= N
			cor[i][j] = x
			cor[j][i] = x
	return cor

def comp_ip(lamda):
	p = len(lamda)
	lamda_sum = 0
	for x in lamda:
		lamda_sum += x
	# print("lamda_sum", lamda_sum)
	lps = 0
	for i in range(p):
		lps += lamda[i]
		print(i+1, lps)
		# print(f"{lps:.3f}/{lamda_sum:.3f}={lps/lamda_sum}>0.95")
		if (lps - lamda[i]) / lamda_sum > 0.95:
		# if lps / lamda_sum > 0.95:
			# print(f"{lps:.3f}/{lamda_sum:.3f}")
			return lps
	return 0

y = covar(y)
ip = comp_ip(lamda)

print(f"I(p)={ip:.3f}")
print("y^1")
for i in range(10):
	if i == 9:
		print(f"{t[i][0]:.3f}*x^{i+1}", end=" ")
	else:
		print(f"{t[i][0]:.3f}*x^{i+1}+", end=" ")
print()
print("y^2")
for i in range(10):
	if i == 9:
		print(f"{t[i][1]:.3f}*x^{i+1}", end=" ")
	else:
		print(f"{t[i][1]:.3f}*x^{i+1}+", end=" ")
print()


print("TEST")
p = 4
corel_test = [
	[1.00, 0.42, 0.54, 0.66],
	[0.42, 1.00, 0.32, 0.44],
	[0.54, 0.32, 1.00, 0.22],
	[0.66, 0.44, 0.22, 1.00]
]
t_test = yakobi(corel_test)
lamda_test = [0] * p
for i in range(p):
	lamda_test[i] = corel_test[i][i]

print("T")
for i in range(p):
	for j in range(p):
		print(f"{t_test[i][j]:.3f}", end="\t")
	print()
print("labda")
print(lamda_test)


# ==================================================
# 2) Проверить гипотезу о значимости коэффициентов корреляции между столбцами матрицы данных.
#print("2) Проверить гипотезу о значимости коэффициентов корреляции между столбцами матрицы данных.")
"""
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
"""

"""
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






corel_test = [
	[1.00, 0.42, 0.54, 0.66],
	[0.42, 1.00, 0.32, 0.44],
	[0.54, 0.32, 1.00, 0.22],
	[0.66, 0.44, 0.22, 1.00]
]
"""
