def console_output(message):
    print(message)


def file_output(message):
    with open('output.txt', 'w') as file:
        file.writelines(message)
