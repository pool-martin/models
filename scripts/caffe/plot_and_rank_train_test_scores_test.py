#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys, os
import argparse

################## Static ARGS - BEGIN ##############

#~ SOLVER_NAME = 'solver.prototxt'
#~ ARCH = 'train_val.prototxt'
#~ LOGFILE = 'nohup.out'
#~ OUTPUT = 'loss.png'

#~ SOLVER_NAME = 'my_quick_solver_finetune.prototxt'
#~ ARCH = 'my_train_val_finetune.prototxt'
#~ LOGFILE = 'logs/my_quick_solver_finetune.txt'
#~ OUTPUT = 'my_quick_solver_finetune.png'

################## Static ARGS - END ##############

def load_args():
	ap = argparse.ArgumentParser(description='Prints learning information: Test/Train ACC, Test/Train losses. Prints a ranking of the tops Test Accuracies.')
	ap.add_argument("-d", "--directory", 
			help="base directory from which search the definition files \
			(default: models/bvlc_googlenet/raw_frames/defs/)",
			default="models/bvlc_googlenet/raw_frames/defs/")
	ap.add_argument("-s", "--solver", 
			help="solver filename",
			type=str, required=True)
	ap.add_argument("-m", "--model", 
			help="model filename",
			type=str, required=True)
	ap.add_argument("-l", "--log", 
			help="log file",
			type=str, required=True)
	#~ ap.add_argument("-ts", "--train-scores-file", 
			#~ help="file containning scores of training",
			#~ type=str, required=True)
	ap.add_argument("-ts", "--train-scores-file", 
			help="file containning scores of training",
			default=None,
			type=str, required=False)
	ap.add_argument("-t", "--training-size", 
			help="size of the training set (default: 100000)", 
			type=int, default=100000)
	ap.add_argument("-r", "--rank-size", 
			help="size of the rakned list. Show X top test accuracies", 
			type=int, default=10)
	ap.add_argument("-i", "--iteration-print", 
			help="use this parameter for printing iteration information also", 
			action='store_true', default=False)
	ap.add_argument("-lr", "--lr-print", 
			help="use this parameter for printing the learning rate also", 
			action='store_true', default=False)
	ap.add_argument("-o", "--output",
			help='output file. Ex: charts/solver.png',
			#~ help='path to the output directory',
			type=str, required=True)

	args = ap.parse_args()
	
	print "Arguments: ", args
	
	return args

"""
	This directory passed as parameter must contain the train_val.prototxt, solver.prototxt, and the nohup.out files.
	The solver must have the paremeters max_iter, display, test_interval, and snapshot.
	nohup.out contain the output that will be read to get the loss and the accuracy.
"""

def getParams(d, s):
	solver_lines = open(d + '/' + args.solver,'rb').readlines()
	for solver_line in solver_lines:
		at = solver_line.split('#')[0].split(':')
		if at[0] == 'max_iter':
			max_iter = int(at[-1].strip())
		if at[0] == 'display':
			display = int(at[-1].strip())
		if at[0] == 'test_interval':
			test_interval = int(at[-1].strip())
		if at[0] == 'snapshot':
			snapshot = int(at[-1].strip())
		# TODO add line for retrieving which archtecture it uses?


	train_val_lines = open(d + '/' + args.model)
	for train_val_line in train_val_lines:
		at = train_val_line.split('#')[0].split(':')
		if at[0].strip() == 'batch_size':
			batch_size = int(at[-1].strip())
			break

	epochs = max_iter*batch_size/s
	iter_by_epoch = int(np.ceil(1.*s/batch_size))
	return max_iter, batch_size, epochs, display, test_interval, snapshot, iter_by_epoch

def getLoss(d, max_iter, batch_size, epochs, display, test_interval, snapshot):
	log = open(d + '/' + args.log,'rb').readlines()
	print "len log before clean: %d" % len(log)

	# Clean log from lines "Data layer prefetch queue empty"
	log = [ line for line in log if "Data layer prefetch queue empty" not in line and "Waiting for data" not in line and "Restarting data prefetching from start" not in line ]
	
	#print log
	print "len log: %d" % len(log)

	# Look for the first iteration
	for i in range(len(log)):
		if 'Iteration 0' in log[i]: break

	print "Iteration 0 line: %d" % i
	
	# Look for the display-th iteration (basically the second display)
	for ii in range(i, len(log)):
		if 'Iteration ' + str(display) in log[ii]: break

	last_snapshot = 0 		# The iteration in which occurred the last snapshot
	last_test = 0 			# The iteration in which occurred the last test
	#secure_skip = ii - i 	# Number of lines between two loss displays
	secure_skip = 5
	len_train = 3			# Number of lines use to print train results
	len_test = 6 			# Number of lines use to print test results
							# Test net output #0: loss1/loss1 = 0.970285 (* 0.3 = 0.291085 loss)
							# Test net output #1: loss1/top-1 = 0.731624
							# Test net output #2: loss2/loss1 = 1.18772 (* 0.3 = 0.356315 loss)
							# Test net output #3: loss2/top-1 = 0.733068
							# Test net output #4: loss3/loss3 = 1.42514 (* 1 = 1.42514 loss)
							# Test net output #5: loss3/top-1 = 0.729332

	train_it = []
	train_loss = []
	train_lr = []
	test_it = []
	test_loss = []
	acc = []
	
	print "Iteration %s line: %d" %( str(display), ii )

	for iii in reversed(range(10)):
		#~ ##~ print log[-(iii+1)]
		sys.stdout.write(log[-(iii+1)])
	#~ print log[-20:]
	#~ print "len(log):",len(log)
	print "************************************"
	#~ print "curr_it:",curr_it
	print "display:",display
	print "last_test:",last_test
	print "test_interval:",test_interval
	print "i:",i
	print "len_test:",len_test
	print "len(log):",len(log)
	print "************************************"
	found = False
	while i < len(log) - 2: # Original Value
	#~ while i < len(log) - 20: # Value set because of error where it tries to split a line without Iteration on it, probably due to being the last snapshot and it having no relation to the epoch

		try:
			if 'Iteration' in log[i]:
				found = True
				l = log[i].split('Iteration')
				print "L: %s" %l
				curr_it = int( l[1].split('(')[0].strip() )
				
				## Loss as the weighted sum from the losses output
				curr_loss = float( l[1].split('loss = ')[1].strip() )
				
				## Loss as the output from loss3/loss3
				#~ curr_loss = float( log[i+len_train].split('=')[1].split('(')[0].strip() ) 
				print "it: %d | loss %f" %(curr_it, curr_loss)
				# Insert in two lists
				train_it.append(curr_it)
				train_loss.append(curr_loss)
			else:
				found = False
				print log[i]
				#i += 1
		#~ except IndexError and ValueError:
		except (IndexError, ValueError):
			sys.stderr.write("Warning: Unable to process correctly the line " + str(i + 1) + ":\n")
			 #~ + '-' + str(i+len_test + 5) + ":\n")
			#~ sys.stderr.write("\t"+str(i+len_test -1 ) + '-' + str(i+len_test + 5) + "\n")
			
			for index,line in enumerate(log[i-3 : i + 3]):
				line_number = i - 4 + index
				sys.stderr.write("\t"+str(line_number)+': '+line)

		if found:
			# Skip secure_skip lines
			i += secure_skip

			# Jump two lines if a snapshot is coming
			#if curr_it + display >= last_snapshot + snapshot:
			if i < len(log):
				if 'Snapshotting to' in log[i]:
					i += 2
					last_snapshot = last_snapshot + snapshot

			# When occurs the last iteration, the loss is in just one line before the test
			if curr_it + display == max_iter:
				l = log[i].split('Iteration')
				
				curr_it = int( l[1].split(',')[0].strip() )
				#curr_loss = float( l[1].split('loss = ')[1].strip() )
				curr_loss = float( log[i+len_train].split('=')[1].split('(')[0].strip() ) 

				# Insert in two lists
				train_it.append(curr_it)
				train_loss.append(curr_loss)

				i += 1

			# Check line-by-line to get the test loss
			print "************************************"
			print "curr_it:",curr_it
			print "display:",display
			print "last_test:",last_test
			print "test_interval:",test_interval
			print "i:",i
			print "len_test:",len_test
			print "len(log):",len(log)
			print "************************************"
			
			if ((curr_it + display >= last_test + test_interval and i+len_test < len(log)) or (i < len(log) and "Testing net" in log[i])):
				
				print "Getting test data"
				
				# Workaround made when there is a train loss line lost before the Test
				try:
					while "Testing net" not in log[i]:
						i += 1
					
					for iii in range(7):
						print log[i + iii]
					
					l = log[i].split('Iteration')
					curr_it = int( l[1].split(',')[0].strip() )
					
					#~ if curr_it < max_iter:
					accuracy = float( log[i+len_test].split('=')[-1].strip() ) 					
					curr_loss = float( log[i+len_test-1].split('=')[1].split('(')[0].strip() )
					
					# when use two lines for accuracy
					# accuracy = float( log[i+len_test-1].split('=')[-1].strip() )
					# curr_loss = float( log[i+len_test-2].split('=')[1].split('(')[0].strip() )
					
					# to avoid 
					if i+len_test+5 < len(log):
						curr_lr = float( log[i+len_test+5].split('=')[-1].strip() )
				#~ except IndexError and ValueError:
				except (IndexError, ValueError):
					sys.stderr.write("Warning: Unable to process correctly the lines " + str(i+len_test -1 ) + '-' + str(i+len_test + 5) + ":\n")
					#~ sys.stderr.write("\t"+str(i+len_test -1 ) + '-' + str(i+len_test + 5) + "\n")
					
					for index,line in enumerate(log[i+len_test -1 : i+len_test + 5]):
						line_number = i + len_test - 1 + index
						sys.stderr.write("\t"+str(line_number)+': '+line)
						#~ sys.stderr.write("\t"+line)
					
					#~ sys.stderr.write("\t"+log[i+len_test])
					#~ sys.stderr.write("\t"+log[i+len_test-1])
					#~ sys.stderr.write("\t"+log[i+len_test+5])
					
					#~ print log[i]
				else:
					if i+len_test+5 < len(log):
						train_lr.append(curr_lr)
					else:
						#~ Repeating curr_lr
						train_lr.append(curr_lr)

					# Insert in two lists
					test_it.append(curr_it)
					test_loss.append(curr_loss)
					acc.append(accuracy)

				last_test = last_test + test_interval
				# Skip secure_skip lines
				i += len_test + 1
			
	# Remove the last item of train (it is including the top-1 of the test)
	return train_it[:-1], train_loss[:-1], test_it, test_loss, acc, train_lr

def getTrainScores(scores_file):
	iters = []
	losses = []
	accs = []
		
	if scores_file is not None:
		with open(scores_file) as input_file:
			scores = input_file.read().split('\n')
		
		if scores[-1] == '':
			scores = scores[:-1] 
		
		for iter_idx in range(0,len(scores),3):
			iter = int(scores[iter_idx].split('_')[-1].split('.')[0])
			loss = float(scores[iter_idx + 1].split(' ')[-2])
			acc = float(scores[iter_idx + 2].split(' ')[-1])
			
			iters.append(iter)
			losses.append(loss)
			accs.append(acc)
	
	return iters, losses, accs

def plot(directory, it, loss, acc, train_lr = None, iter_by_epoch = None):
	def flat_list(_list):
		## Ravel has trouble if lists have different sizes. In this case, if train and test doest not have the same number of accs.
		## Therefore this flat workaround it is necessary
		return [item for sublist in _list for item in sublist]
		
	if train_lr is not None:
		fig, axes = plt.subplots(nrows=2, ncols=1)

		# Needs a bigger image for fitting both the charts
		fig.set_figheight(10)
		fig.set_figwidth(8)
		
		ax1 = axes[0]
		ax_lr = axes[1]
	else:
		fig, ax1 = plt.subplots()
	
	#ax1.plot(it[0], acc[0], 'b', linewidth = 2, label = "train acc", linestyle='-.')# 'g-.')
	ax1.plot(it[1], acc[1], 'r', linewidth = 2, label = "test acc", linestyle='-.')# 'g-.')
	
	## Ravel has trouble if lists have different sizes. In this case, if train and test doest not have the same number of accs.
	#~ ax1.set_yticks(np.arange(round(min(np.ravel(acc)),2) - 0.01, round(max(np.ravel(acc)),2) + 0.01, 0.01))
	
	ax1.set_yticks(np.arange(round(min(flat_list(acc)),2) - 0.01, round(max(flat_list(acc)),2) + 0.01, 0.01))
	#~ ax1.set_yticks(np.arange(0, 1.1, 0.1)) # Not very good range
	#~ ax1.set_yticks(np.arange(0.75, 1.01, 0.01))
	#~ ax1.set_yticks(np.arange(0.75, 0.95, 0.01))
	ax1.set_ylabel('Accuracy', color='g')
	for tl in ax1.get_yticklabels():
		tl.set_color('g')
	
	## For Epochs view pt1
	epoch_step = 10
	
	ax1.set_xlabel('Epoch')
	plt.xticks(np.arange(min(it[1]), max(it[1]) + epoch_step, epoch_step))
	#~ plt.xticks(np.arange(min(it[1]), max(it[1]) + epoch_step, epoch_step), rotation=17)
	#~ plt.xticks(np.arange(min(it[0]), max(it[0])+1, epoch_step))
	#~ plt.xticks(np.arange(0, 26, epoch_step))
	##
	
	## For iterations view pt1
	#~ ax1.set_xlabel('Iteration')
	#~ # plt.xticks(np.arange(min(it[0]), max(it[0])+40, 40))
	#~ plt.xticks(np.arange(min(it[0]), max(it[0])+42350, 42350), rotation=17) # Five Epochs
	##
	
	#plt.gca().xaxis.grid(True)
	plt.grid()

	ax2 = ax1.twinx()
	ax2.plot(it[0], loss[0], 'b', linewidth = 1.5, label = "train loss", linestyle='-')
	ax2.plot(it[1], loss[1], 'r', linewidth = 1.5, label = "test loss", linestyle='-')
	#~ ax2.set_yticks(np.arange(0, 5.5, 0.5)) # Not very good range
	#~ ax2.set_yticks(np.arange(0, 2.0, 0.2))
	#~ ax2.set_yticks(np.arange(0.35, 0.80, 0.05))
	#~ ax2.set_yticks(np.arange(0.35, 0.65, 0.05))
	ax2.set_ylabel('Loss', color='y')
	for tl in ax2.get_yticklabels():
		tl.set_color('y')
	
	## For Epochs+Iterations view pt1
	if iter_by_epoch is not None:
		ax3 = ax2.twiny()
		ax3.set_xlabel('Iteration')
		#~ plt.xticks(np.arange(min(it[0]), max(it[0])+1, 5), np.arange(min(it[0])*8470, max(it[0])*8470+5*8470, 5*8470), rotation=17)
		plt.xticks(np.arange(min(it[0]), max(it[0])+1, epoch_step), np.arange(min(it[0])*iter_by_epoch, max(it[0])*iter_by_epoch + epoch_step*iter_by_epoch, epoch_step*iter_by_epoch), rotation=17)
		
		#~ TODO Update here for always having the ticks based on the other xticks
		
		## 8470 is the epoch size
	##

	if iter_by_epoch is None:
		plt.title('Loss and Accuracy by Epoch')
	else:
		plt.title('Loss and Accuracy by Epoch/Iteration', y=1.13)
		#~ plt.gca().tight_layout()
		plt.gcf().subplots_adjust(top=0.86)
	#~ ax1.legend(loc = 2, borderaxespad=0.4, prop={'size':12})
	#~ ax2.legend(loc = 1, borderaxespad=0.4, prop={'size':12})
	ax1.legend(bbox_to_anchor=(0., 1.03, 1., .103), loc = 2, borderaxespad=0.4, prop={'size':12})
	ax2.legend(bbox_to_anchor=(0., 1.03, 1., .103), loc = 1, borderaxespad=0.4, prop={'size':12})
	
	"""
	"""
	if train_lr is not None:
		ax_lr.plot(it[0], train_lr, 'g', linewidth = 2, label = "learning rate", linestyle='-')
		
		ax_lr.set_ylim(train_lr[-1]*0.9, train_lr[0]*1.1)
		
		plt.xticks(np.arange(min(it[1]), max(it[1]) + epoch_step, epoch_step))
		ax_lr.set_xlabel('Epoch')
		
		ax_lr.set_ylabel('Learning Rate', color='g')
	#~ """
		
		
	
	plt.savefig(directory + '/' + args.output)
	plt.show()


args = load_args()
directory = args.directory
size = args.training_size

print "Path to Log:"
print "\t",args.directory +'/'+ args.log
print "Path to train scores file:"
print "\t",args.train_scores_file

max_iter, batch_size, epochs, display, test_interval, snapshot, iter_by_epoch = getParams(directory, size)

print "%d | %d | %d | %d | %d | %d | %d |" % (max_iter, batch_size, epochs, display, test_interval, snapshot, iter_by_epoch)
train_it, train_loss, test_it, test_loss, test_acc, train_lr = getLoss(directory, max_iter, batch_size, epochs, display, test_interval, snapshot)
#print "%s | %s | %s | %s | %s | %s |" % (train_it, train_loss, test_it, test_loss, test_acc, train_lr)

#train_it, train_loss, train_acc = getTrainScores(args.train_scores_file)
train_acc = []

## For Epochs view pt2 AND For Epochs+Iterations view pt2
#~ iter_by_epoch2= max_iter/epochs ## This was used by Ramon, but got replaced by iter_by_epoch from getParams

train_epochs = [ iteration/iter_by_epoch for iteration in train_it]

#~ train_it = [ v/iter_by_epoch for v in train_it]
#~ train_it = [ v/iter_by_epoch - 1 for v in train_it] # Hand made align of chart
#~ train_it = [ v/iter_by_epoch + 1 for v in train_it] # Hand made align of chart

#~ test_it = [ v/iter_by_epoch for v in test_it]
#~ test_it = [ v/iter_by_epoch + 1 for v in test_it] # Hand made align of chart
#~ test_epochs = [ iteration/iter_by_epoch + 1 for iteration in test_it] # Hand made align of chart, unnecessary after fixing corrext test_it extraction
test_epochs = [ iteration/iter_by_epoch for iteration in test_it]
##

###

iter_by_epoch = None if not args.iteration_print else iter_by_epoch
#~ train_lr = None if not args.lr_print else train_lr
### "Workaround" for when getting a train_it plus
train_lr = None if not args.lr_print else train_lr[:len(train_epochs)]

print "train_epochs \n"
print train_epochs
print "test_epochs \n"
print test_epochs
print "iter_by_epoch \n"
print iter_by_epoch

plot(directory, [train_epochs, test_epochs], [train_loss, test_loss], [train_acc, test_acc], train_lr, iter_by_epoch)

### Ranking

ranked_indexes = np.array(test_acc).argsort()[-args.rank_size:][::-1]

print "Top", args.rank_size, "Test Accuracies"
#~ print "Test ACC |\t Train ACC |\t Iteration |\t Epoch "
print "Test ACC |  Train ACC  |  Iteration  |  Epoch "

for index in ranked_indexes:
	#~ print test_acc[index], '\t', train_acc[index], '\t', test_it[index]
	if index < len(train_acc):
		#~ print test_acc[index], '\t', train_acc[index], '\t', train_it[index], '\t\t', train_epochs[index]
		print "  {0:.2f}%".format(test_acc[index] * 100), '    ', "{0:.2f}%".format(train_acc[index] * 100), '     ', train_it[index], '      ', train_epochs[index]
	else:
		#~ print test_acc[index], '\t', "MISSING", '\t', test_it[index], '\t\t', test_epochs[index]
		#~ print test_acc[index], '\t', "MISSING", '\t', test_it[index], '\t\t', test_epochs[index]
		print "  {0:.2f}%".format(test_acc[index] * 100), '    ', "MISSING", '    ', test_it[index], '      ', test_epochs[index]


