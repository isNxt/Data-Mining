with open('../data/kosarak/kosarak.dat', 'r') as input, open('../data/kosarak/kosarak.txt', 'w') as output:
    for transaction_str in input:
        transaction_str = transaction_str.strip('\n')
        transaction = transaction_str.split(' ')
        for item in set(transaction):
            if item[0] == ' ':
                new_item = item.replace(' ', '', 1)
                output.write(new_item + ',')
            else:
                output.write(item + ',')
        output.write('\n')


