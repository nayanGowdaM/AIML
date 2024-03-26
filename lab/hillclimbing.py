def hilClimbing(func, start, step_size=0.1, max_iter=10000):
    CPos=start
    CVal=func(CPos)

    for i in range( max_iter):
        print(f"{i}. {CPos} -> {CVal}")
        NPPos=CPos + step_size
        NPVal=func(NPPos)

        NNPos=CPos - step_size
        NNVal=func(NNPos)

        if NPVal>=CVal and NPVal >=NNVal:
            CPos=NPPos
            CVal=NPVal
        elif NNVal>+CVal and NNVal>+NPVal:
            CPos=NNPos
            CVal=NNVal
        else:
            break
    return CPos, CVal


if __name__=="__main__":
    while True:
        funcStr=input("Enter function in x: ")
        try:
            x=0
            eval(funcStr)
            break
        except Exception as e:
            print(f"Invalid  expression.Try again\nError:{e}")

    func = lambda x: eval(funcStr)

    while True:
        startStr=input("Enter the start value  of x : " )
        try:
            start= float(startStr)
            break
        except Exception as e:
            print(f"Error:{e}")

    
    pos, maxima = hilClimbing(func, start, 0.1,1000)
    print(f"Maxima of {funcStr} is {maxima} at {pos}")