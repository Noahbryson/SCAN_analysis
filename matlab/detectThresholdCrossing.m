function X = detectThresholdCrossing(data, threshold, distance)

X= [];
i = 1;
while(true)
    if data(i) > threshold
        X = [X; i];
        i = i + distance;
    else
        i = i+1;
    end
    if i == length(data)
        break
    end
end
end