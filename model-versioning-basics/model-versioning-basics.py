def promote_model(models: list) -> str:
    """
    Returns the model name as a string.
    """
    maxPricedItem = max(models, key=lambda x:x['accuracy'])
    best_model_v = maxPricedItem['name']
    
    updatedModels = []
    for model in models:
        if model['accuracy'] >= maxPricedItem['accuracy']:
            updatedModels.append(model)
    
    maxPricedItem = min(updatedModels, key=lambda x:x['latency'])

    updatedModels2 = []
    for model in updatedModels:
        if model['latency'] <= maxPricedItem['latency']:
            updatedModels2.append(model)
    
    maxPricedItem = max(updatedModels2, key=lambda x:x['timestamp'])
    best_model_v = maxPricedItem['name']
    
    
    return best_model_v