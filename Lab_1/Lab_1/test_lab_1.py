import math
import numpy as np

def test_function(function, input, output, kind = 'array'):

	result = True
	l = len(input)

	input = [function(i) for i in input]

	for i in range(l):

		if kind == 'scalar':
			check = input[i] == output[i]
		if kind == 'array':
			check = (input[i] == output[i]).all()

		if check:
			pass
		else:
			print('Your function did NOT pass all the test cases')
			result = False
			break

	if result == True: 
		print('All test cases are pass.')

def test_function_2(function, input, output):

	result = True
	l = len(input)

	input = [function(i[0], i[1]) for i in input]

	for i in range(l):

		check = input[i] == output[i]


		if check:
			pass
		else:
			print('Your function did NOT pass all the test cases')
			result = False
			break

	if result == True: 
		print('All test cases are pass.')



def test_basic_sigmoid(basic_sigmoid):

	test_input = [1, 0, -2]
	test_output = [0.7310585786300049, 0.5, 0.11920292202211755]

	test_function(basic_sigmoid, test_input, test_output, kind = 'scalar')

def test_sigmoid(sigmoid):

	test_input = [
					np.array([-1, 0, 1]),

					np.array([[1, 2, 3], 
							 [4, 5, 6]]),

					np.array([[3, 5, 2, 4],
						      [7, 6, 8, 8],
						      [1, 6, 7, 7]])					
					]
	test_output = [
					np.array([0.2689414213699951, 0.5, 0.7310585786300049]),

					np.array([[0.7310585786300049, 0.8807970779778823, 0.9525741268224334],
       					     [0.9820137900379085, 0.9933071490757153, 0.9975273768433653]]),

					np.array([[0.9525741268224334, 0.9933071490757153, 0.8807970779778823, 0.9820137900379085],
       					     [0.9990889488055994, 0.9975273768433653, 0.9996646498695336,0.9996646498695336],
       					     [0.7310585786300049, 0.9975273768433653, 0.9990889488055994,0.9990889488055994]])
					]

	test_function(sigmoid, test_input, test_output)

def test_sigmoid_derivative(sigmoid_derivative):

	test_input = [
					np.array([-1, 0, 1]), 
					np.array([[1, 2, 3], 
							 [4, 5, 6]]),
					np.array([[3, 5, 2, 4],
						      [7, 6, 8, 8],
						      [1, 6, 7, 7]])
									
					]
	test_output = [
					np.array([0.19661193324148185, 0.25 , 0.19661193324148185]),

					np.array([[0.19661193324148185 , 0.10499358540350662 , 0.045176659730912   ],
       						 [0.017662706213291107, 0.006648056670790033, 0.002466509291359931]]),

					np.array([[0.045176659730912     , 0.006648056670790033  ,0.10499358540350662   , 0.017662706213291107  ],
       					  [0.000910221180121784  , 0.002466509291359931  ,0.00033523767075636815, 0.00033523767075636815],
       					  [0.19661193324148185   , 0.002466509291359931  ,0.000910221180121784  , 0.000910221180121784  ]])
					]

	
	test_function(sigmoid_derivative, test_input, test_output)

def test_image2vector(image2vector):

	test_input = [
					np.array([[[10, 11, 12], [13, 14, 15], [16, 17, 18]],
               				 [[20, 21, 22], [23, 24, 25], [26, 27, 28]],
               				 [[30, 31, 32], [33, 34, 35], [36, 37, 38]]]),
					np.zeros((5,2,3))					
					]

	test_output = [
					np.array([[10],[11],[12],[13],[14],[15],[16],[17],[18],[20],[21],[22],[23],[24],[25],[26],[27],[28],[30],[31],[32],[33],[34],[35],[36],[37],[38]]),

					np.array([[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]])


					]


	test_function(image2vector, test_input, test_output)


def test_normalize_rows(normalize_rows):

	test_input = [
					np.array([[1, -2, 3], 
							 [-4, 5, 6]]),
					np.identity((5))				
					]

	test_output = [
					np.array([[ 0.2672612419124244, -0.5345224838248488,  0.8017837257372732],
       						 [-0.4558423058385518,  0.5698028822981898,  0.6837634587578276]]),

					np.array([[1., 0., 0., 0., 0.],
       						 [0., 1., 0., 0., 0.],
       						 [0., 0., 1., 0., 0.],
       						 [0., 0., 0., 1., 0.],
       						 [0., 0., 0., 0., 1.]])
	]

	test_function(normalize_rows, test_input, test_output)


def test_softmax(softmax):

	test_input = [
					np.array([[1, -2, 3], 
							 [-4, 5, 6]]),
					np.array([[0,1,-10, 5]])				
					]

	test_output = [
					np.array([[1.1849965453500957e-01, 5.8997504019027806e-03,8.7560059506308763e-01],
       						 [3.3188906581985211e-05, 2.6893249549828524e-01,7.3103431559513277e-01]]),

					np.array([[6.573261223679088e-03, 1.786797653804133e-02, 2.984255978654958e-07, 9.755584638126817e-01]])
				]

	test_function(softmax, test_input, test_output)

def test_L1(L1):

	test_input = [
					[np.array([1,2,3]), np.array([4,5,-10])],
					[np.zeros((5,)), np.array([5, 3, 1, -1, -3])]
				]			

	test_output = [19, 13]
	
	test_function_2(L1, test_input, test_output)

def test_L2(L2):

	test_input = [
				 [np.array([1,2,3]), np.array([4,5,-10])],
				 [np.zeros((5,)), np.array([5, 3, 1, -1, -3])]
	]
			
	
	test_output = [187, 45]
	test_function_2(L2, test_input, test_output)















