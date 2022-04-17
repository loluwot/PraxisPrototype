from itertools import accumulate
import itertools



def formatting(txt):
    return ['{}: {} to {}'.format(*([v := x.split('.\t')[1].strip().split(' Comment'), v[0], v[1].strip().split('\t')[0], v[1].strip().split('\t')[-1]][1:])) for x in txt.split('\n')]


questions = [] 
s = 'Overall reaction: '
xs = '''terrible											wonderful	
difficult											easy	
frustrating											satisfying	
inadequate power											adequate power	
dull											stimulating	
rigid											flexible'''

questions.append(['Overall reaction: {} to {}'.format(*(l.split())) for l in xs.split('\n')])
xss = ['''7.	Reading characters on the screen Comment	hard											easy	
8.	Highlighting simplifies task Comment	not at all											very much	
9.	Organization of information Comment	confusing											very clear	
10.	Sequence of screens Comment	confusing											very clear''',
'''11.	Use of terms throughout system Comment	inconsistent											consistent	
12.	Terminology related to task Comment	never											always	
13.	Position of messages on screen Comment	inconsistent											consistent	
14.	Prompts for input Comment	confusing											clear	
15.	Computer informs about its progress Comment	never											always	
16.	Error messages Comment	unhelpful											helpful''',
'''17.	Learning to operate the system Comment	difficult											easy	
18.	Exploring new features by trial and error Comment	difficult											easy	
19.	Remembering names and use of commands Comment	difficult											easy	
20.	Performing tasks is straightforward Comment	never											always	
21.	Help messages on the screen Comment	unhelpful											helpful	
22.	Supplemental reference materials Comment	confusing											clear''',
'''23.	System speed Comment	too slow											fast enough	
24.	System reliability Comment	unreliable											reliable	
25.	System tends to be Comment	noisy											quiet	
26.	Correcting your mistakes Comment	difficult											easy	
27.	Designed for all levels of users Comment	never											always''']
question_names = ['OVERALL REACTION TO THE SOFTWARE', 'SCREEN', 'TERMINOLOGY AND SYSTEM INFORMATION', 'LEARNING', 'SYSTEM CAPABILITIES']
questions.extend([formatting(y) for y in xss])

lens = list(accumulate(questions, lambda x, y: x + len(y), initial=0))
total_questions = list(itertools.chain.from_iterable(questions))
print('\n'.join(total_questions))
# zip(question_names, questions)
print(lens)
print('QUESTIONS',questions)
def process_answers(ss):
    LL = [(z := [x.split('=')[1] for x in s.split('\n') if not any([sz in x for sz in ['positive', 'negative']])]) + ['N/A']*(27 - len(z)) for s in ss]
    LS = [[L[lens[i]:lens[i+1]] for i in range(len(lens) - 1)] for L in LL]
    print(list(map(lambda *args: list(map(lambda tup: ','.join(tup), zip(*args))), questions, *LS)))
    return '\n'.join(list(map(lambda tup: '\n'.join([f'{tup[0]}{","*len(ss)}'] + tup[1]), zip(question_names, map(lambda *args: list(map(lambda tup: ','.join(tup), zip(*args))), questions, *LS)))))

s = '''q1=7
q2=7
q3=7
q4=7
q5=7
q6=7
q7=5
q8=5
q9=5
q10=5
q11=9
q12=8
q13=9
q14=9
q15=9
q16=9
q17=9
q18=9
q19=9
q20=7
q21=8
q22=5
q23=9
q24=9
q25=9
q26=5
q27=7
negative1=Overlapping words
positive1=Clear Terminology'''
s2 = '''q1=8
q2=8
q3=8
q4=8
q5=8
q6=8
q7=7
q8=8
q9=8
q10=8
q11=8
q12=8
q13=8
q14=8
q15=8
q16=8
q17=8
q18=8
q19=8
q20=8
q21=7
q22=-1
q23=8
q24=8
q25=8
q26=8
q27=8
negative1=Drop Down can be clearer
negative2=Navigation to following pages can be clearer
positive1=Meets the purpose and tasks
positive2=Ease to use'''

s3 = '''q1=5
q2=6
q3=6
q4=7
q5=6
q6=6
q7=6
q8=4
q9=5
q10=4
q11=5
q12=5
q13=5
q14=6
q15=6
q16=6
q17=6
q18=4
q19=6
negative1=Must use clearer and simpler terminology
negative2=Too many screen changes
positive1=Simple, large fonts used'''

s4 ='''q1=9
q2=8
q3=9
q4=9
q5=8
q6=9
q7=9
q8=9
q9=9
q10=9
q11=9
q12=9
q13=7
q14=8
q15=7
q16=6
q17=9
q18=9
q19=9
q20=8
q21=8
q22=4
q23=8
q24=8
q25=7
q26=6
q27=9'''

open('test_answer.csv', 'w').write(process_answers([s, s2, s3, s4]))