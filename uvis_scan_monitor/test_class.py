class Scan1:
	def __init__(self, filepath):
		self.filepath = filepath
		print(self.filepath)

class Scan2():
	def __init__(self, filepath):
		self.filepath = filepath
		print(self.filepath)

#class Scan3(filepath):
#	def __init__(filepath):
#		self.filepath = filepath
#		print(self.filepath)

#class Scan4(filepath):
#	def __init__:
#		self.filepath = filepath


if __name__ == '__main__':
	filepath = input('Please enter a file path.\n')

	try:
		t1 = Scan1(filepath)
		print('test 1 succeeded')
	except Exception as e:
		print(f'test 1 failed with error {e}:\n\tclass Scan1:\n\t\tdef__init__(filepath)')

	try:
		t2 = Scan2(filepath)
		print('test 2 succeeded')
	except Exception as e:
		print(f'test 2 failed with error {e}:\n\tclass Scan2():\n\t\tdef__init__(filepath)')

#	try:
#		t3 = Scan3(filepath)
#		print('test 3 succeeded')
#	except Exception as e:
#		print(f'test 3 failed with error {e}:\n\tclass Scan3(filepath):\n\t\tdef__init__(filepath)')

#	try:
#		t4 = Scan4(filepath)
#		print('test 4 succeeded')
#	except Exception as e:
#		print(f'test 4 failed with error {e}:\n\tclass Scan4(filepath):\n\t\tdef__init__:')


