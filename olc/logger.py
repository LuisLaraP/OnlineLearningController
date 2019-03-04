class Logger:

	def __init__(self, filename):
		self.file = open(filename, 'w')

	def close(self):
		self.file.close()

	def setNames(self, names):
		self.file.write('\t'.join(names) + '\n')
