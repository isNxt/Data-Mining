with open('../data/transactions/transactions.csv', 'r') as input, open('../data/transactions/transactions.txt', 'w') as output:
    column_str = input.readline()
    column_str = column_str.strip('\n')
    column = list(filter(None, column_str.split(',')))
    print(column)
    print(column[0])
    print(column[1])

    for line in input.readlines():
        line = line.strip('\n')
        line_data = list(filter(None, line.split(',')))
        print(line_data)
        index = 0
        for item in line_data:
            if item == '1':
                output.write(column[index] + ',')
            index += 1
        output.write('\n')
