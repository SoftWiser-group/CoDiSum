import json
from pygments import highlight
from pygments.lexers import JavaLexer
from pygments.formatters import RawTokenFormatter
import numpy as np
import re


def word_like(min_freq):
    vc = json.loads(open('/home/kingxu/commitgen/ASE_commitgen_data/msg.vocab.count.json').read())
    # print(vc)
    x = list()
    y = list()
    z = list()
    k = dict()
    k['<eos>'], k['<start>'], k['<unk>'], k['<copy>'] = 0, 1, 2, 3
    num = 4
    sum1, sum2, sum3 = 0, 0, 0
    for i, j in vc.items():
        if i.isalpha() and i.islower() and j >= min_freq:
            print(i)
            x.append(i)
            k[i] = num
            num += 1
            sum1 += j
        elif is_not(i):
            x.append(i)
            k[i] = num
            num += 1
        elif i.isalpha():
            y.append(i)
            sum2 += j
        elif i.isalnum():
            z.append(i)
            sum3 += j
    print(len(x), len(y), len(z))
    print(sum1, sum2, sum3)
    # json.dump(k, open('msg_vocab.json', 'w'))
    # print(y)
    # print(z)


def is_not(word):
    nt = {"don't", "doesn't", "can't", "shouldn't", "isn't", "didn't", "won't", "wasn't", "aren't", "wouldn't",
          "hasn't", "weren't", "couldn't", "haven't"}
    if word in nt:
        return True
    return False


def find_diff(all_java=True, one_block=False):
    l = json.load(open('/home/kingxu/commitgen/ASE_commitgen_data/ASEdata_filterd_100.json', 'r'), encoding='utf-8')
    vc = json.loads(open('msg_vocab.json').read())
    full = list()
    for i in l:
        diff = i['diffs']
        files = diff.count('diff --git')
        java_files = diff.count('.java') // 4
        blocks = diff.count('@@') // 2
        if all_java and (files < 1 or files != java_files):
            continue
        if one_block and blocks != 1:
            continue
        ls = diff.splitlines()
        tokens, names, attributes, classes, functions = list(), list(), list(), list(), list()
        for j in ls:
            t, n, a, c, f = pygment_one_line(j)
            tokens += t
            names += n
            attributes += a
            classes += c
            functions += f
        names = set(names)
        attributes = set(attributes)
        classes = set(classes)
        functions = set(functions)
        vd = gen_variable_dict(names, attributes, classes, functions)
        msg = i['msgs'].strip('. \t\r\n').split(' ')
        msg = [i for i in msg if i != '' and not i.isspace()]
        # print(msg)
        msg_tokens = list()
        for k in msg:
            if k in vd:
                msg_tokens.append((3, find_variable_loc(k, tokens)))
            elif k.lower() in vc:
                msg_tokens.append((vc[k.lower()], list()))
            else:
                msg_tokens.append((2, list()))
        # print(msg_tokens)
        d = dict()
        d['diff'] = tokens
        d['msg'] = msg_tokens
        d['msg_text'] = msg
        d['variable'] = vd
        full.append(d)
    print(len(full))
    if one_block:
        filename = 'oneblock_dataset.json'
    else:
        filename = 'dataset.json'
    json.dump(full, open(filename, 'w'))


def gen_variable_dict(nameset, attributesset, classset, functionset):
    d = dict()
    for n, i in enumerate(nameset):
        d[i] = 'n{}'.format(n)
    for n, i in enumerate(attributesset):
        d[i] = 'a{}'.format(n)
    for n, i in enumerate(classset):
        d[i] = 'c{}'.format(n)
    for n, i in enumerate(functionset):
        d[i] = 'f{}'.format(n)
    # print(d)
    return d


def find_variable_loc(variable, tokens):
    locs = list()
    for n, i in enumerate(tokens):
        if variable == i[1]:
            locs.append(n)
    return locs


def pygment_one_line(linestring):
    l = list()
    namelist = list()
    attributelist = list()
    classlist = list()
    functionlist = list()
    if len(linestring) < 1 or linestring.startswith('+++') or linestring.startswith('---'):
        return l, namelist, attributelist, classlist, functionlist
    st = linestring[0]
    # print(st)
    linestring = linestring[1:].strip()
    if linestring == '':
        return l, namelist, attributelist, classlist, functionlist
    if st == '@':
        l.append((2, '<NewBlock>'))
        linestring = linestring[linestring.find('@@') + 3:].strip()
        if linestring == '':
            return l, namelist, attributelist, classlist, functionlist
        cls = 2
    elif st == ' ':
        cls = 2
    elif st == '-':
        cls = 1
    elif st == '+':
        cls = 3
    else:
        return l, namelist, attributelist, classlist, functionlist
    if linestring.startswith('/*') or linestring.startswith('*') or linestring.endswith('*/'):
        l.append((cls, 'JAVADOC'))
        return l, namelist, attributelist, classlist, functionlist
    x = highlight(linestring, JavaLexer(), RawTokenFormatter())
    x = str(x, encoding='utf-8')
    for y in x.splitlines():
        ys = y.split('\t')
        print(ys)
        s = eval(ys[1]).strip(' \t\n\r')
        if s != '':
            # print(ys)
            if "Token.Literal.Number.Float" == ys[0]:
                l.append((cls, 'FLOAT'))
            elif "Token.Literal.Number.Integer" == ys[0]:
                l.append((cls, 'INTEGER'))
            elif "Token.Literal.Number.Hex" == ys[0]:
                l.append((cls, 'HEX'))
            elif "Token.Literal.String" == ys[0]:
                l.append((cls, 'STRING'))
            elif "Token.Literal.String.Char" == ys[0]:
                l.append((cls, 'CHAR'))
            elif "Token.Name.Namespace" == ys[0]:
                l.append((cls, 'NAMESPACE'))
            elif "Token.Comment.Single" == ys[0]:
                l.append((cls, 'SINGLE'))
            elif "Token.Comment.Multiline" == ys[0]:
                l.append((cls, 'MULTILINE'))
            elif 'Token.Name.Decorator' == ys[0]:
                l.append((cls, 'DECORATOR'))
            elif 'Token.Name' == ys[0]:
                namelist.append(s)
                l.append((cls, s))
            elif 'Token.Name.Attribute' == ys[0]:
                attributelist.append(s)
                l.append((cls, s))
            elif 'Token.Name.Class' == ys[0]:
                classlist.append(s)
                l.append((cls, s))
            elif 'Token.Name.Function' == ys[0]:
                functionlist.append(s)
                l.append((cls, s))
            else:
                l.append((cls, s))
    # print(l)
    return l, namelist, attributelist, classlist, functionlist


def pygment_mul_line(java_lines):
    string = '\n'.join(java_lines)
    if string == '':
        return list(), dict()
    x = highlight(string, JavaLexer(), RawTokenFormatter())
    x = str(x, encoding='utf-8')
    tokenList = list()
    variableDict = dict()
    nameNum, attNum, clsNum, fucNum = 0, 0, 0, 0
    otherDict = dict()
    floatNum, numberNum, strNum = 0, 0, 0
    for y in x.splitlines():
        ys = y.split('\t')
        # print(ys)
        s = eval(ys[1])
        if s == '\n':
            tokenList.append('<nl>')
        elif s == 'NewBlock':
            tokenList.append('<nb>')
        elif s.isspace():
            lines = s.count('\n')
            for _ in range(lines):
                tokenList.append('<nl>')
        elif "Token.Literal.Number.Float" == ys[0]:
            if s not in otherDict:
                sT = 'FLOAT{}'.format(floatNum)
                otherDict[s] = sT
                floatNum += 1
            tokenList.append(otherDict[s])
        elif ys[0].startswith('Token.Literal.Number'):
            if s not in otherDict:
                sT = 'NUMBER{}'.format(numberNum)
                otherDict[s] = sT
                numberNum += 1
            tokenList.append(otherDict[s])
        elif ys[0].startswith('Token.Literal.String'):
            if s not in otherDict:
                sT = 'STRING{}'.format(strNum)
                otherDict[s] = sT
                strNum += 1
            tokenList.append(otherDict[s])
        elif "Token.Name.Namespace" == ys[0]:
            tokenList.append('NAMESPACE')
        elif "Token.Comment.Single" == ys[0]:
            tokenList.append('SINGLE')
            tokenList.append('<nl>')
        elif "Token.Comment.Multiline" == ys[0]:
            lines = s.count('\n')
            for _ in range(lines):
                tokenList.append('COMMENT')
                tokenList.append('<nl>')
            tokenList.append('COMMENT')
        elif 'Token.Name.Decorator' == ys[0]:
            tokenList.append('@')
            tokenList.append(s[1:].lower())
        elif 'Token.Name' == ys[0]:
            if s not in variableDict:
                sT = 'n{}'.format(nameNum)
                variableDict[s] = sT
                nameNum += 1
            tokenList.append(s)
        elif 'Token.Name.Attribute' == ys[0]:
            if s not in variableDict:
                sT = 'a{}'.format(attNum)
                variableDict[s] = sT
                attNum += 1
            tokenList.append(s)
        elif 'Token.Name.Class' == ys[0]:
            if s not in variableDict:
                sT = 'c{}'.format(clsNum)
                variableDict[s] = sT
                clsNum += 1
            tokenList.append(s)
        elif 'Token.Name.Function' == ys[0]:
            if s not in variableDict:
                sT = 'f{}'.format(fucNum)
                variableDict[s] = sT
                fucNum += 1
            tokenList.append(s)
        else:
            a = s.splitlines()
            for i in a:
                if i != '' and not i.isspace():
                    tokenList.append(i)
                tokenList.append('<nl>')
            tokenList.pop()
    return tokenList, variableDict


def split_variable(var):
    pattern1 = re.compile(r'[a-zA-Z]+')
    pattern2 = re.compile(r'[a-zA-Z][a-z]*')
    words = pattern1.findall(var)
    wordList = list()
    for i in words:
        if i.islower() or i.isupper():
            wordList.append(i.lower())
        else:
            for j in pattern2.findall(i):
                wordList.append(j.lower())
    # print(wordList)
    return wordList


def data_constraint(min_diff, max_diff, min_msg, max_msg):
    data = json.load(open('oneblock_dataset.json', 'r'), encoding='utf-8')
    condata = list()
    for i in data:
        diff = i['diff']
        msg = i['msg']
        if min_diff <= len(diff) < max_diff and min_msg <= len(msg) < max_msg:
            condata.append(i)
    print(len(condata))
    json.dump(condata, open('oneblock_dataset_{}_{}_{}_{}.json'.format(min_diff, max_diff, min_msg, max_msg), 'w'))


def gen_diff_vocab():
    # 调用一次，用于生成diff的词典就行
    data = json.load(open('dataset_1_400_3_20.json', 'r'), encoding='utf-8')
    k = dict()
    k['<eos>'], k['<start>'], k['<unk>'], k['<copy>'] = 0, 1, 2, 3
    num = 4
    for i in data:
        diff = i['diff']
        variable = i['variable']
        for d in diff:
            if d[1] in variable:
                if variable[d[1]] not in k:
                    k[variable[d[1]]] = num
                    num += 1
            else:
                if d[1] not in k:
                    k[d[1]] = num
                    num += 1
    print(k)
    json.dump(k, open('diff_vocab.json', 'w'))


def variable_replace():
    data = json.load(open('oneblock_dataset_1_400_3_20.json', 'r'), encoding='utf-8')
    dv = json.loads(open('diff_vocab.json').read())
    for i in data:
        diff = i['diff']
        new_diff = list()
        variable = i['variable']
        for d in diff:
            if d[1] in variable:
                if variable[d[1]] in dv:
                    new_diff.append((d[0], dv[variable[d[1]]]))
                else:
                    new_diff.append((d[0], 2))
            else:
                if d[1] in dv:
                    new_diff.append((d[0], dv[d[1]]))
                else:
                    new_diff.append((d[0], 2))
        i['diff'] = new_diff
        i['diff_text'] = diff
    json.dump(data, open('oneblock_dataset_1_400_3_20_digit.json', 'w'))


def remove_unk():
    data = json.load(open('oneblock_dataset_1_400_3_20_digit.json', 'r'), encoding='utf-8')
    d = list()
    for i in data:
        msg = i['msg']
        is_unk = False
        # if msg[0][0] == 25 and msg[1][0] == 476 and (msg[2][0] == 1185 or msg[2][0] == 835):
        #     is_unk = True
        for m in msg:
            if m[0] == 2 or m[0] == 1185 or m[0] == 835:
                is_unk = True
                break
        if not is_unk:
            d.append(i)
    print(len(d))
    json.dump(d, open('oneblock_dataset_1_400_3_20_digit_nounk.json', 'w'))


def gen_tensor4json(filename, start, end):
    data = json.load(open(filename, 'r'), encoding='utf-8')
    data = data[start: end]
    length = end - start
    d_sign = np.zeros([length, 400])
    d_cont = np.zeros([length, 400])
    m_cont = np.zeros([length, 21])
    m_locs = np.zeros([length, 20, 400])
    for idx1, i in enumerate(data):
        for idx2, c in enumerate(i['diff']):
            d_sign[idx1, idx2] = c[0]
            d_cont[idx1, idx2] = c[1]
        m_cont[idx1, 0] = 1
        for idx2, c in enumerate(i['msg']):
            m_cont[idx1, idx2 + 1] = c[0]
            for idx3 in c[1]:
                m_locs[idx1, idx2, idx3] = 1
    # print(d_sign, d_cont, m_cont, m_locs)
    return d_sign, d_cont, m_cont, m_locs, data


if __name__ == '__main__':
    # word_like(8)
    # find_diff(one_block=True)
    # gen_variable_dict({'x','y'},{'a','b'})
    # data_constraint(1,400,3,20)
    pygment_one_line(' public class ImmutableDomainObjectSet<T> extends AbstractSet<T> implements DomainObjectSet<T> {')
    # splitVariable('IOMap')
    # x = highlight('', JavaLexer(), RawTokenFormatter())
    # x = str(x, encoding='utf-8')
    # print(x)
    # variable_replace()
    # remove_unk()
    # gen_tensor4json('oneblock_dataset_1_400_3_20_digit_nounk.json', 0, 10000)
