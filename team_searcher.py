from datetime import datetime
def team_search(team):
    team = team.split()

    sentences = []

    with open('teams.txt', 'r') as file:
        for line in file:
            sentences.append(line.strip())
    check = 0
    for sentence in sentences:
        for word in team:
            check = 1
            temp = sentence.split()
            if word not in temp:
                check = 0
                break
        if check == 1:
            return sentence


print(team_search('TFT11_Malphite TFT11_Garen TFT11_Neeko TFT11_Lux TFT11_Illaoi TFT11_Syndra TFT11_Sylas TFT11_Galio '))

date = ('_' + str(datetime.now().day) + '/' + str(datetime.now().month) + '/' + str(datetime.now().year) + '_' +
        str(datetime.now().hour) + ':' + str(datetime.now().minute).zfill(2))
print(date)