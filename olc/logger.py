class Logger:

	def __init__(self, filename, names):
		self.file = open(filename, 'w')
		self.file.write('\t'.join(names) + '\n')

	def close(self):
		self.file.close()
