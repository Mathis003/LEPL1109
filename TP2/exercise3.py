# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



def exo3(flag_lambda_0): 
	
	beta_true = [1.5, 2.4, 0, 1]  # True synthetic model

	n_point = 80

	X = np.random.rand(n_point, 1)  # Features
	N = (np.random.rand(n_point, 1) - 0.5) * 0.5  # Additive noise
	y = np.polyval(beta_true, X) + N  # Noisy target

	_X, X_test, _y, y_test = train_test_split(X, y, test_size=0.1)  # 10% test set

	X_train, X_val, y_train, y_val = train_test_split(
		_X, _y, test_size=0.366
	)  # 0.33% validation set


	n_train = X_train.shape[0]
	n_test = X_test.shape[0]
	n_val = X_val.shape[0]


	X_plot = np.linspace(0, 1, 200)

	# Plot the data with color discrimination for each set
	plt.figure(figsize=(10, 5))
	plt.scatter(X_train, y_train, color="blue", label="train")
	plt.scatter(X_val, y_val, color="green", label="val")
	plt.scatter(X_test, y_test, color="red", label="test")
	plt.legend()


	# Vandermonde matrix
	V_train = np.vander(np.reshape(X_train, n_train), increasing=False)

	lamb_points = np.logspace(3, -3, num=7)  # 6 different lambda values

	if flag_lambda_0:
		lamb_points = np.append(0, lamb_points)
	colors = [mpl.cm.coolwarm(x) for x in lamb_points]


	val_err_points = []
	train_err_points = []
	val_err_line = []
	train_err_line = []

	# First plot: vizualization of interpolation polynomial
	# Loop through the 6 lambda values, compute the regularized model and plot the interpolation polynomial on [0, 1]
	for (l, c) in zip(lamb_points, colors):
		beta = (
			np.linalg.inv(V_train.T @ V_train + l * np.eye(n_train, n_train))
			@ V_train.T
			@ y_train
		)
		plt.plot(X_plot, np.polyval(beta, X_plot), c=c)
		if l == 1.0:
			dx = 0.07
			dy = -0.1
		elif l == 0.1:
			dx = 0.07
			dy = +0.1
		else:
			dx = 0.16
			dy = 0
		plt.text(X_plot[-1] + dx, np.polyval(beta, X_plot[-1]) + dy, r"$\lambda$ = %s" % l)

		val_err_points.append(
			np.linalg.norm(np.polyval(beta, X_val) - y_val) / np.sqrt(n_val)
		)
		train_err_points.append(
			np.linalg.norm(np.polyval(beta, X_train) - y_train) / np.sqrt(n_train)
		)


	# Second plot: vizualization of training and testing errors

	lamb_line = np.logspace(3, -3, 100)  # More lambda values, used to plot error functions

	for l in lamb_line:
		beta = (
			np.linalg.inv(V_train.T @ V_train + l * np.eye(n_train, n_train))
			@ V_train.T
			@ y_train
		)
		val_err_line.append(
			np.linalg.norm(np.polyval(beta, X_val) - y_val) / np.sqrt(n_val)
		)
		train_err_line.append(
			np.linalg.norm(np.polyval(beta, X_train) - y_train) / np.sqrt(n_train)
		)


	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Polynomial models $f_4$ and train-validation-test points")
	plt.show()

	plt.figure(figsize=(10, 5))
	plt.plot(lamb_line, val_err_line, color="r", label="Validation error")
	plt.plot(lamb_line, train_err_line, color="b", label="Training error")

	if not flag_lambda_0:
		plt.xscale("log")

	plt.scatter(lamb_points, val_err_points, color="r")
	plt.scatter(lamb_points, train_err_points, color="b")


	l_opt = lamb_line[np.argmin(val_err_line)]

	print("Optimal lambda is", l_opt)

	beta_opt = (
		np.linalg.inv(V_train.T @ V_train + l_opt * np.eye(n_train, n_train))
		@ V_train.T
		@ y_train
	)
	test_err = np.linalg.norm(np.polyval(beta_opt, X_test) - y_test) / np.sqrt(n_test)


	plt.plot(
		[l_opt, l_opt],
		[np.min(train_err_line), np.max(val_err_line)],
		linestyle="dashed",
		color="green",
		label="$\lambda^*$",
	)
	plt.plot(
		[min(lamb_line), max(lamb_line)],
		[test_err, test_err],
		linestyle="dotted",
		color="purple",
		label="Test error with $\lambda^*$",
	)

	plt.legend()

	plt.xlabel("$\lambda$")
	plt.ylabel("Error")
	plt.title("Error of $f_4$ on the validation set for varying $\lambda$")
	plt.show()