class Logger:

	def __init__(self, filename):
		self.file = open(filename, 'w')

	def close(self):
		self.file.close()

	def log(self, values):
		strs = [str(x) for x in values]
		self.file.write('\t'.join(strs) + '\n')

	def setNames(self, names):
		self.file.write('\t'.join(names) + '\n')
