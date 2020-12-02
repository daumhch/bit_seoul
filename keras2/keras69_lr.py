weight = 0.5
input = 0.5
goal_prediction = 0.8
lr = 1 # defalut 0.01
# lr이 너무 작으면(예를들어 0.0001), 1000번 돌 때까지 목표치에 도달하지 못한다 
# (가까워지다 range(1000)이 끝난다)

# lr이 너무 크면(예를들어 1), 1000번 돌 때까지 목표치에 도달하지 못한다 
# (왔다리 갔다리 반복)


for iteration in range(1000):
    prediction = input*weight
    error = (prediction - goal_prediction)**2
    print("iteration:",iteration," Error:",str(error)+"\tPrediction:"+str(prediction))

    up_prediction = input*(weight+lr)
    up_error = (goal_prediction - up_prediction)**2

    down_prediction = input*(weight-lr)
    down_error = (goal_prediction - down_prediction)**2

    if(down_error<up_error):
        weight = weight-lr
    if(down_error>up_error):
        weight = weight+lr


