import json
from data_process_tools import pygment_mul_line, split_variable
import numpy as np
import random
import re


class Data4CopynetV3:
    def __init__(self):
        # initial all variable
        # save msg data as list
        self.msgtext = []
        # save splited msg data as list
        self.msg = []
        # save diff generated code change as list
        self.difftext = []
        # split diff and save as list
        self.difftoken = []
        # split diff variable as diff attribution
        self.diffatt = []
        # + or - mark before a token or a line,
        # when it's length is smaller than diff token, it's marking a line.
        # when they're equal, it's marking a token
        self.diffmark = []
        # dict for both diff token and msg word
        self.word2index = {}
        # if a word can't be generated, set it's genmask to 0
        self.genmask = []
        # if a word can't be copy, set it's copymask to 0
        self.copymask = []
        # save the dict for entity and it's representation
        self.variable = []
        # save the word index of the first word that only appear in msg
        self.difftoken_start = 0

    def build(self, jsonfile, all_java, one_block):
        # this function read data in json file and fill msg*, diff*, variable
        # jsonfile is file name of a json file. string
        # all_java means only java code change. boolean
        # one_block means only one diff block. boolean
        data = json.load(open(jsonfile, 'r'), encoding='utf-8')
        pattern = re.compile(r'\w+')    # for splitting msg
        # for i in data:
        for x, i in enumerate(data):
            if x > 1000000000:  # x for debug, set value of x to a small num
                break
            diff = i['diffs']
            files = diff.count('diff --git')
            java_files = diff.count('.java') // 4
            blocks = diff.count('@@') // 2
            if all_java and (files < 1 or files != java_files):
                continue
            if one_block and blocks != 1:
                continue
            # print(diff)
            ls = diff.splitlines()
            java_lines = list()
            diff_marks = list()
            other_file = False
            for line in ls:
                if len(line) < 1:
                    continue
                if line.startswith('+++') or line.startswith('---'):
                    if not line.endswith('.java'):
                        other_file = True
                        break
                    continue
                st = line[0]
                line = line[1:].strip()
                if st == '@':
                    java_lines.append('NewBlock ' + line[line.find('@@') + 3:].strip())
                    diff_marks.append(2)
                elif st == ' ':
                    if line.startswith('/*') or line.startswith('*') or line.endswith('*/'):
                        java_lines.append('COMMENT')
                    else:
                        java_lines.append(line)
                    diff_marks.append(2)
                elif st == '-':
                    if line.startswith('/*') or line.startswith('*') or line.endswith('*/'):
                        java_lines.append('COMMENT')
                    else:
                        java_lines.append(line)
                    diff_marks.append(1)
                elif st == '+':
                    if line.startswith('/*') or line.startswith('*') or line.endswith('*/'):
                        java_lines.append('COMMENT')
                    else:
                        java_lines.append(line)
                    diff_marks.append(3)
            if other_file:
                continue
            tokenList, varDict = pygment_mul_line(java_lines)
            msg = pattern.findall(i['msgs'])
            msg = [i for i in msg if i != '' and not i.isspace()]
            self.msgtext.append(i['msgs'])
            self.msg.append(msg)
            self.difftext.append(diff)
            # length of diff token and diff mark aren't equal
            self.difftoken.append(tokenList)
            self.diffmark.append(diff_marks)
            self.variable.append(varDict)
        # self.save_data(True, True, True, True, True, False)

    def re_split_msg(self):
        # to split msg in case build function are wrong
        msgs = list()
        pattern = re.compile(r'\w+')
        for i in self.msgtext:
            msg = pattern.findall(i)
            msg = [j for j in msg if j != '' and not j.isspace()]
            msgs.append(msg)
        self.msg = msgs
        # self.save_data(False, False, False, True, False, False)

    def re_process_diff(self):
        # split variable in diff and save it into diff attribution
        # this function should be invoked after build only once
        diff_tokens, diff_marks, diff_atts = [], [], []
        for i, j, k in zip(self.difftoken, self.diffmark, self.variable):
            diff_token, diff_att = [], []
            for x in i:
                if x in k:
                    diff_att.append(split_variable(x))
                    diff_token.append(x)
                else:
                    diff_att.append([])
                    diff_token.append(x)
            diff_mark, diff_token, diff_att = self.mark_token(j, diff_token, diff_att)
            # translate line mark into token mark
            diff_tokens.append(diff_token)
            diff_marks.append(diff_mark)
            diff_atts.append(diff_att)
        self.diffmark = diff_marks
        self.difftoken = diff_tokens
        self.diffatt = diff_atts
        # self.save_data(False, False, True, False, False, False)

    def mark_token(self, marklist, tokenlist, attlist):
        lineNum = 0
        diff_mark = list()
        for i in tokenlist:
            diff_mark.append(marklist[lineNum])
            if i == '<nl>':
                lineNum += 1
        while lineNum < len(marklist):
            diff_mark.append(marklist[lineNum])
            tokenlist.append('<nl>')
            attlist.append([])
            lineNum += 1
        return diff_mark, tokenlist, attlist

    def filter(self, min_diff, max_diff, min_msg, max_msg):
        na, nb, nc, nd, ne, nf, ng = [], [], [], [], [], [], []
        for i, j, k, l, m, n, o in zip(self.difftoken, self.diffmark, self.msg, self.variable, self.difftext,
                                       self.msgtext, self.diffatt):
            diff = []
            for idx, d in enumerate(i):
                if (d == '<nb>' or d == '<nl>') and idx + 1 < len(j) and j[idx + 1] != 2:
                    diff.append(d)
                    if j[idx + 1] == 1:
                        diff.append('-')
                    else:
                        diff.append('+')
                else:
                    diff.append(d)
            # print(diff)
            if min_diff <= len(diff) < max_diff and min_msg <= len(k) < max_msg:
                save = True
                for w in k:
                    # if (not w.isalpha() and w not in l) or w.lower() == 'flaw' or w.lower() == 'flaws':
                    if not w.isalpha() and w not in l:
                        save = False
                        break
                if save:
                    na.append(i), nb.append(j), nc.append(k), nd.append(l), ne.append(m), nf.append(n), ng.append(o)
        self.difftoken, self.diffmark, self.msg, self.variable, self.difftext = na, nb, nc, nd, ne
        self.msgtext, self.diffatt = nf, ng

    def shuffle(self):
        random.seed(1)
        random.shuffle(self.difftoken)
        random.seed(1)
        random.shuffle(self.diffmark)
        random.seed(1)
        random.shuffle(self.msg)
        random.seed(1)
        random.shuffle(self.variable)
        random.seed(1)
        random.shuffle(self.difftext)
        random.seed(1)
        random.shuffle(self.msgtext)
        random.seed(1)
        random.shuffle(self.diffatt)

    def build_word2index1(self):
        self.word2index = {'<eos>': 0, '<start>': 1, '<unkm>': 2}
        num = 3
        # vp = re.compile(r'(n|a|f|c)\d+$')
        # cp = re.compile(r'(FLOAT|NUMBER|STRING)\d+$')
        v_set = set()
        word_count1, word_count2 = dict(), dict()
        lemmatization = json.load(open('lemmatization.json'))
        for i, j, k in zip(self.difftoken, self.msg, self.variable):
            for x in j:
                if x in k:
                    continue
                x = x.lower()
                if x in lemmatization:
                    x = lemmatization[x]
                if x in word_count2:
                    word_count2[x] += 1
                else:
                    word_count2[x] = 1
            for x in i:
                if x in k:
                    v_set.add(k[x])
                    continue
                if x in word_count1:
                    word_count1[x] += 1
                else:
                    word_count1[x] = 1
        word_count1 = sorted(word_count1.items(), key=lambda x: x[1], reverse=True)
        word_count2 = sorted(word_count2.items(), key=lambda x: x[1], reverse=True)
        for i in word_count2:
            if num >= 10000:
                break
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in v_set:
            if i not in self.word2index:
                self.word2index[i] = num
                num += 1
        self.difftoken_start = num
        self.word2index['<unkd>'] = num
        num += 1
        for i in word_count1:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1

    def gen_tensor1(self, start, end, vocab_size, diff_len=200, msg_len=20):
        lemmatization = json.load(open('lemmatization.json'))
        length = end - start
        msg = self.msg[start: end]
        diff = self.difftoken[start: end]
        diff_m = self.diffmark[start: end]
        va = self.variable[start: end]
        d_mark = np.zeros([length, diff_len])
        d_word = np.zeros([length, diff_len])
        mg = np.zeros([length, msg_len + 1])
        for i, (j, k, l, m) in enumerate(zip(diff, diff_m, msg, va)):
            for idx, (dt, dm) in enumerate(zip(j, k)):
                d_mark[i, idx] = dm
                dt = m[dt] if dt in m else dt
                dn = self.word2index[dt] if dt in self.word2index else self.word2index['<unkd>']
                d_word[i, idx] = dn
            mg[i, 0] = 1
            for idx, c in enumerate(l):
                c = m[c] if c in m else c.lower()
                c = lemmatization[c] if c in lemmatization else c
                c0 = self.word2index[c] if c in self.word2index else self.word2index['<unkm>']
                c0 = self.word2index['<unkm>'] if c0 >= self.difftoken_start else c0
                mg[i, idx + 1] = c0
        genmask = np.zeros([vocab_size, ])
        copymask = np.zeros([vocab_size, ])
        genmask[:10000] = 1
        copymask[:self.difftoken_start] = 1
        return d_mark, d_word, mg, genmask, copymask

    def build_word2index2(self):
        # this function should be invoked after re_process_diff
        # but as later as possible
        self.word2index = {'<eos>': 0, '<start>': 1, '<unkm>': 2}
        num = 3
        # count word frequencies in diff and msg
        word_count1, word_count2, word_count3 = dict(), dict(), dict()
        v_set = set()
        lemmatization = json.load(open('lemmatization.json'))
        for i, j, k, l in zip(self.difftoken, self.msg, self.diffatt, self.variable):
            for x in j:
                if x in l:
                    continue
                x = x.lower()
                x = lemmatization[x] if x in lemmatization else x
                word_count2[x] = word_count2[x] + 1 if x in word_count2 else 1
            for x in i:
                if x in l:
                    v_set.add(l[x])
                    continue
                word_count1[x] = word_count1[x] + 1 if x in word_count1 else 1
            for x in k:
                for y in x:
                    word_count3[y] = word_count3[y] + 1 if y in word_count3 else 1
        word_count1 = sorted(word_count1.items(), key=lambda x: x[1], reverse=True)
        word_count2 = sorted(word_count2.items(), key=lambda x: x[1], reverse=True)
        word_count3 = sorted(word_count3.items(), key=lambda x: x[1], reverse=True)
        for i in word_count2:
            if num >= 10000:
                break
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in v_set:
            if i not in self.word2index:
                self.word2index[i] = num
                num += 1
        self.difftoken_start = num
        self.word2index['<unkd>'] = num
        num += 1
        for i in word_count1:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in word_count3:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        # self.save_data(False, False, False, False, False, True)
        f = open('kv.txt', 'w')
        for i in word_count1:
            f.write('{}\t\t\t{}\n'.format(i[0], i[1]))
        for i in word_count2:
            f.write('{}\t\t\t{}\n'.format(i[0], i[1]))
        for i in word_count3:
            f.write('{}\t\t\t{}\n'.format(i[0], i[1]))
        # json.dump(self.word2index, open('data4mul_block/word2index.json', 'w'))

    def gen_tensor2(self, start, end, vocab_size, diff_len=200, attr_num=5, msg_len=20):
        lemmatization = json.load(open('lemmatization.json'))
        length = end - start
        msg = self.msg[start: end]
        diff = self.difftoken[start: end]
        diff_m = self.diffmark[start: end]
        diff_a = self.diffatt[start: end]
        va = self.variable[start: end]
        d_mark = np.zeros([length, diff_len])
        d_word = np.zeros([length, diff_len])
        d_attr = np.zeros([length, diff_len, attr_num])
        mg = np.zeros([length, msg_len + 1])
        for i, (j, k, l, m, n) in enumerate(zip(diff, diff_m, msg, va, diff_a)):
            for idx, (dt, dm, da) in enumerate(zip(j, k, n)):
                d_mark[i, idx] = dm
                dt = m[dt] if dt in m else dt
                dn = self.word2index[dt] if dt in self.word2index else self.word2index['<unkd>']
                d_word[i, idx] = dn
                for idx2, a in enumerate(da):
                    if idx2 >= attr_num:
                        break
                    d_attr[i, idx, idx2] = self.word2index[a] if a in self.word2index else self.word2index['<unkd>']
            mg[i, 0] = 1
            for idx, c in enumerate(l):
                c = m[c] if c in m else c.lower()
                c = lemmatization[c] if c in lemmatization else c
                c0 = self.word2index[c] if c in self.word2index else self.word2index['<unkm>']
                c0 = self.word2index['<unkm>'] if c0 >= self.difftoken_start else c0
                mg[i, idx + 1] = c0
        genmask = np.zeros([vocab_size, ])
        copymask = np.zeros([vocab_size, ])
        genmask[:10000] = 1
        copymask[:self.difftoken_start] = 1
        return d_mark, d_word, d_attr, mg, genmask, copymask

    def build_word2index3(self):
        # this function should be invoked after re_process_diff
        # but as later as possible
        self.word2index = {'<eos>': 0, '<start>': 1, '<unkm>': 2}
        num = 3
        # count word frequencies in diff and msg
        word_count1, word_count2, word_count3 = dict(), dict(), dict()
        v_set = set()
        lemmatization = json.load(open('lemmatization.json'))
        for i, j, k, l in zip(self.difftoken, self.msg, self.diffatt, self.variable):
            for x in j:
                if x in l:
                    continue
                x = x.lower()
                x = lemmatization[x] if x in lemmatization else x
                word_count2[x] = word_count2[x] + 1 if x in word_count2 else 1
            for x in i:
                if x in l:
                    v_set.add(l[x])
                    continue
                word_count1[x] = word_count1[x] + 1 if x in word_count1 else 1
            for x in k:
                for y in x:
                    word_count3[y] = word_count3[y] + 1 if y in word_count3 else 1
        word_count1 = sorted(word_count1.items(), key=lambda x: x[1], reverse=True)
        word_count2 = sorted(word_count2.items(), key=lambda x: x[1], reverse=True)
        word_count3 = sorted(word_count3.items(), key=lambda x: x[1], reverse=True)
        for i in word_count2:
            if num >= 10000:
                break
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in v_set:
            if i not in self.word2index:
                self.word2index[i] = num
                num += 1
        self.difftoken_start = num
        self.word2index['<unkd>'] = num
        self.word2index['<sp>'] = num + 1
        self.word2index['<add>'] = num + 2
        self.word2index['<del>'] = num + 3
        num += 4
        for i in word_count1:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1
        for i in word_count3:
            if i[0] not in self.word2index:
                self.word2index[i[0]] = num
                num += 1

    def gen_tensor3(self, start, end, vocab_size, diff_len=200, msg_len=20):
        lemmatization = json.load(open('lemmatization.json'))
        length = end - start
        msg = self.msg[start: end]
        diff = self.difftoken[start: end]
        diff_m = self.diffmark[start: end]
        diff_a = self.diffatt[start: end]
        va = self.variable[start: end]
        d_mark = np.zeros([length, diff_len])
        d_word = np.zeros([length, diff_len])
        d_attr = np.zeros([length, diff_len])
        mg = np.zeros([length, msg_len + 1])
        for i, (j, k, l, m, n) in enumerate(zip(diff, diff_m, msg, va, diff_a)):
            num = 0
            for idx, (dt, dm, da) in enumerate(zip(j, k, n)):
                d_mark[i, idx] = dm
                dt = m[dt] if dt in m else dt
                dn = self.word2index[dt] if dt in self.word2index else self.word2index['<unkd>']
                d_word[i, idx] = dn
                if len(da) > 0:
                    if num + len(da) + 1 >= diff_len:
                        break
                    if dm == 2:
                        d_attr[i, num] = self.word2index['<sp>']
                    elif dm == 1:
                        d_attr[i, num] = self.word2index['<del>']
                    else:
                        d_attr[i, num] = self.word2index['<add>']
                    num += 1
                    for a in da:
                        d_attr[i, num] = self.word2index[a] if a in self.word2index else self.word2index['<unkd']
                        num += 1
            mg[i, 0] = 1
            for idx, c in enumerate(l):
                c = m[c] if c in m else c.lower()
                c = lemmatization[c] if c in lemmatization else c
                c0 = self.word2index[c] if c in self.word2index else self.word2index['<unkm>']
                c0 = self.word2index['<unkm>'] if c0 >= self.difftoken_start else c0
                mg[i, idx + 1] = c0
        genmask = np.zeros([vocab_size, ])
        copymask = np.zeros([vocab_size, ])
        genmask[:10000] = 1
        copymask[:self.difftoken_start] = 1
        return d_mark, d_word, d_attr, mg, genmask, copymask

    def save_data(self, version, save_difftext=False, save_msgtext=False, save_diff=False, save_msg=False,
                  save_variable=False, save_word2index=False):
        # you can do it any time, just save all data
        if save_difftext:
            json.dump(self.difftext, open('data4CopynetV3/difftextV{}.json'.format(version), 'w'))
        if save_msgtext:
            json.dump(self.msgtext, open('data4CopynetV3/msgtextV{}.json'.format(version), 'w'))
        if save_diff:
            json.dump(self.difftoken, open('data4CopynetV3/difftokenV{}.json'.format(version), 'w'))
            json.dump(self.diffmark, open('data4CopynetV3/diffmarkV{}.json'.format(version), 'w'))
            json.dump(self.diffatt, open('data4CopynetV3/diffattV{}.json'.format(version), 'w'))
        if save_msg:
            json.dump(self.msg, open('data4CopynetV3/msgV{}.json'.format(version), 'w'))
        if save_variable:
            json.dump(self.variable, open('data4CopynetV3/variableV{}.json'.format(version), 'w'))
        if save_word2index:
            json.dump(self.word2index, open('data4CopynetV3/word2indexV{}.json'.format(version), 'w'))
            json.dump(self.genmask, open('data4CopynetV3/genmaskV{}.json'.format(version), 'w'))
            json.dump(self.copymask, open('data4CopynetV3/copymaskV{}.json'.format(version), 'w'))
            json.dump(self.difftoken_start, open('data4CopynetV3/numV{}.json'.format(version), 'w'))

    def load_data(self, version, load_difftext=True, load_msgtext=True, load_diff=True, load_msg=True,
                  load_variable=True, load_word2index=True):
        # java load data from disk
        if load_difftext:
            self.difftext = json.load(open('data4CopynetV3/difftextV{}.json'.format(version)))
        if load_msgtext:
            self.msgtext = json.load(open('data4CopynetV3/msgtextV{}.json'.format(version)))
        if load_diff:
            self.difftoken = json.load(open('data4CopynetV3/difftokenV{}.json'.format(version)))
            self.diffmark = json.load(open('data4CopynetV3/diffmarkV{}.json'.format(version)))
            self.diffatt = json.load(open('data4CopynetV3/diffattV{}.json'.format(version)))
        if load_msg:
            self.msg = json.load(open('data4CopynetV3/msgV{}.json'.format(version)))
        if load_variable:
            self.variable = json.load(open('data4CopynetV3/variableV{}.json'.format(version)))
        if load_word2index:
            self.word2index = json.load(open('data4CopynetV3/word2indexV{}.json'.format(version)))
            self.genmask = json.load(open('data4CopynetV3/genmaskV{}.json'.format(version)))
            self.copymask = json.load(open('data4CopynetV3/copymaskV{}.json'.format(version)))
            self.difftoken_start = int(json.load(open('data4CopynetV3/numV{}.json'.format(version))))

    def data_constraint(self, min_diff, max_diff, min_msg, max_msg):
        na, nb, nc, nd, ne, nf, ng = [], [], [], [], [], [], []
        for i, j, k, l, m, n, o in zip(self.diffmark, self.difftoken, self.msg, self.variable, self.difftext,
                                    self.msgtext, self.diffatt):
            if min_diff <= len(i) < max_diff and min_msg <= len(k) < max_msg:
                na.append(i), nb.append(j), nc.append(k), nd.append(l), ne.append(m), nf.append(n), ng.append(o)
        self.diffmark, self.difftoken, self.msg, self.variable, self. difftext, self.msgtext = na, nb, nc, nd, ne, nf
        self.diffatt = ng
        # self.save_data(True, True, True, True, True, False)

    def deduplication(self):
        na, nb, nc, nd, ne, nf, ng = [], [], [], [], [], [], []
        diffset = set()
        for i, j, k, l, m, n, o in zip(self.diffmark, self.difftoken, self.msg, self.variable, self.difftext,
                                       self.msgtext, self.diffatt):
            iii = str(i)+''.join(j)+n if len(k) < 10 else n
            if iii not in diffset:
                na.append(i), nb.append(j), nc.append(k), nd.append(l), ne.append(m), nf.append(n), ng.append(o)
                diffset.add(iii)
        self.diffmark, self.difftoken, self.msg, self.variable, self.difftext, self.msgtext = na, nb, nc, nd, ne, nf
        self.diffatt = ng

    def remove_unk(self):
        na, nb, nc, nd, ne, nf, ng = [], [], [], [], [], [], []
        lemmatization = json.load(open('lemmatization.json'))
        for i, j, k, l, m, n, p in zip(self.diffmark, self.difftoken, self.msg, self.variable, self.difftext,
                                    self.msgtext, self.diffatt):
            is_unk = False
            for o in k:
                if o.lower() in lemmatization:
                    continue
                if (o.lower() not in self.word2index and o not in l) or o.lower() == 'flaw' or o.lower() == 'flaws':
                    is_unk = True
                    break
            if not is_unk:
                na.append(i), nb.append(j), nc.append(k), nd.append(l), ne.append(m), nf.append(n), ng.append(p)
        self.diffmark, self.difftoken, self.msg, self.variable, self.difftext, self.msgtext = na, nb, nc, nd, ne, nf
        self.diffatt = ng
        # self.save_data(True, True, True, True, True, False)

    def gen_tensor(self, start, end, vocab_size, diff_len=300, msg_len=20, attribute_num=5):
        lemmatization = json.load(open('lemmatization.json'))
        length = end - start
        msg = self.msg[start: end]
        difftoken = self.difftoken[start: end]
        diffmark = self.diffmark[start: end]
        diffatt = self.diffatt[start: end]
        variable = self.variable[start: end]
        d_mark = np.zeros([length, diff_len])
        d_word = np.zeros([length, diff_len])
        d_attr = np.zeros([length, diff_len, attribute_num])
        mg = np.zeros([length, msg_len + 1])
        cp = np.zeros([length, msg_len, 2])
        vocab_mask = np.zeros([vocab_size, ])
        for i, (j, k, l, m, n) in enumerate(zip(diffmark, difftoken, msg, variable, diffatt)):
            for idx, (mk, tk, ak) in enumerate(zip(j, k, n)):
                d_mark[i, idx] = mk
                tkn = self.word2index[tk] if tk in self.word2index else 2
                d_word[i, idx] = tkn
                idx1 = 0
                for va in ak:
                    if idx1 >= attribute_num:
                        break
                    if va in self.word2index:
                        d_attr[i, idx, idx1] = self.word2index[va]
                        idx1 += 1
                while idx1 < attribute_num:
                    d_attr[i, idx, idx1] = 7
                    idx1 += 1
            mg[i, 0] = 1
            for idx, c in enumerate(l):
                if c in m:
                    c0 = self.word2index[m[c]] if m[c] in self.word2index else 3
                    cp[i, idx, 1] = 1
                else:
                    c = c.lower()
                    if c in lemmatization:
                        c = lemmatization[c]
                    c0 = self.word2index[c] if c in self.word2index else 3
                    if c0 >= self.msg_word_start:
                        cp[i, idx, 0] = 1
                mg[i, idx + 1] = c0
        vocab_mask[0:len(self.genmask)] = self.genmask
        return d_mark, d_word, d_attr, mg, cp, vocab_mask

    def get_data(self, start, end, re_difftext=False, re_msgtext=False, re_diffmark=False, re_difftoken=False,
                 re_msg=False, re_variable=False):
        data = list()
        if re_difftext:
            data.append(self.difftext[start: end])
        if re_msgtext:
            data.append(self.msgtext[start: end])
        if re_diffmark:
            data.append(self.diffmark[start: end])
        if re_difftoken:
            data.append(self.difftoken[start: end])
        if re_msg:
            data.append(self.msg[start: end])
        if re_variable:
            data.append(self.variable[start: end])
        return data

    def get_word2index(self):
        return self.word2index, self.difftoken_start

    def test(self):
        print(len(self.difftoken))
        # sum1, sum2, num = 0, 0, 0
        # for i, j in zip(self.difftoken, self.msg):
        #     sum1 += len(i)
        #     sum2 += len(j)
        #     if 3 <= len(i) < 251 and 3 <= len(j) < 21:
        #         num += 1
        # print(sum1 / len(self.difftoken), sum2 / len(self.msg), num)


def recall_ref():
    TE_S = 83000  # test_start_index
    TE_E = 90661  # test_end_index
    dataset = Data4CopynetV3()
    dataset.load_data(12)
    msg, variable = dataset.get_data(TE_S, TE_E, re_msg=True, re_variable=True)
    refs = []
    for i, j in zip(msg, variable):
        ref = []
        for x in i:
            if x in j:
                ref.append(x)
        refs.append(ref)
    json.dump(refs, open('recall_ref.json', 'w'))


if __name__ == '__main__':
    # dataset = Data4CopynetV3()
    # 20190120 2208
    # dataset.build('/home/kingxu/commitgen/ASE_commitgen_data/ASEdata_filterd_100.json', True, False)
    # dataset.save_data(0, True, True, True, True, True)
    # 20190121 0152
    # dataset.load_data(0, True, True, True, True, True, False)
    # dataset.filter(3, 201, 3, 21)
    # dataset.shuffle()
    # dataset.re_process_diff()
    # dataset.deduplication()
    # dataset.data_constraint(3, 301, 3, 21)
    # dataset.save_data(10, True, True, True, True, True, True)
    # dataset.test()
    # 20190921 0220
    # dataset.load_data(True, True, True, True, True, True)
    # dataset.build_word2index(700, 4300, 5000)
    # dataset.remove_unk()
    # dataset.build_word2index(700, 4300, 5000)
    # dataset.save_data(1, True, True, True, True, True, True)
    # dataset.test()
    # 20190121 1908
    # dataset.load_data(13, True, True, True, True, True, True)
    # a, b, c, d, e, f = dataset.gen_tensor3(0, 10, 11000)
    # for i, j, k, l in zip(a, b, c, d):
    #     print(i)
    #     print(j)
    #     print(k)
    #     print(l)
    # print(e, f)
    # 20190122 1724
    # dataset.load_data(1, True, True, True, True, True, True)
    # dataset.build_word2index(500, 3500, 4000)
    # dataset.remove_unk()
    # dataset.build_word2index(500, 3500, 4000)
    # dataset.save_data(2, True, True, True, True, True, True)
    # dataset.test()
    # 20190122 1829
    # dataset.load_data(2, True, True, True, True, True, True)
    # dataset.build_word2index(500, 3500, 3000)
    # dataset.remove_unk()
    # dataset.build_word2index(500, 3500, 3000)
    # dataset.save_data(3, True, True, True, True, True, True)
    # dataset.test()
    # 20190122 2147 modify build word2index function
    # dataset.load_data(1, True, True, True, True, True, True)
    # dataset.build_word2index(500, 3500, 3000)
    # dataset.remove_unk()
    # dataset.build_word2index(500, 3500, 3000)
    # dataset.save_data(4, True, True, True, True, True, True)
    # dataset.test()
    # 20190123 0138
    # dataset.load_data(4, True, True, True, True, True, True)
    # dataset.build_word2index(500, 3500, 2500)
    # dataset.remove_unk()
    # dataset.build_word2index(500, 3500, 2500)
    # dataset.save_data(5, True, True, True, True, True, True)
    # dataset.test()
    # 20190127 2300
    # dataset.load_data(10, load_word2index=False)
    # dataset.build_word2index1()
    # dataset.save_data(11, True, True, True, True, True, True)
    # 20190131 1208
    # dataset.load_data(10, load_word2index=False)
    # dataset.build_word2index2()
    # dataset.save_data(12, True, True, True, True, True, True)
    # 201902012 1236
    # dataset.load_data(10, load_word2index=False)
    # dataset.build_word2index3()
    # dataset.save_data(13, True, True, True, True, True, True)
    # 20190221 1241
    # dataset.load_data(10, load_word2index=False)
    # dataset.deduplication()
    # dataset.save_data(10, True, True, True, True, True, True)
    # dataset.test()
    recall_ref()

