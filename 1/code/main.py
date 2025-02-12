from pandas import read_excel
import math

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

# 1a) средние по столбцам, дисперсии по столбцам
print("1a) средние по столбцам, дисперсии по столбцам")
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
print("1б) стандартизированная матрица")
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
print("1в) ковариационная матрица")
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
print("Ковариационная матрица:")
for j in range(p):
	for i in range(p):
		print(f"{covar[j][i]:.3f}\t", end="")
	print()
print()

# 1г) корелляционная матрица
print("1г) корелляционная матрица")
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
print("2) Проверить гипотезу о значимости коэффициентов корреляции между столбцами матрицы данных.")
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
print("Статистика T для каждого коэффицента корелляции:")
for j in range(p):
	for i in range(p):
		if i != j:
			print(f"{T[j][i]:.3f}\t", end="")
		else:
			print("#####\t", end="")
	print()
print()

alpha = 0.05
f = N-2 # 57
print(f"alpha={alpha}, f={f}")
t_table = 2.0024655
print(f"t_table={t_table}")
print()

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
