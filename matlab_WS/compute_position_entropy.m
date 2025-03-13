function total_entropy = compute_position_entropy(probabilities)
    % 计算位置熵的总和
    % 输入：probabilities 是一个包含概率的向量
    % 输出：total_entropy 是位置熵的总和
    
    % 确保概率向量中每个元素都是非负的，并且总和为1
    if any(probabilities < 0) || abs(sum(probabilities) - 1) > 1e-10
        error('概率向量必须包含非负元素，并且总和为 1');
    end
    
    % 计算熵
    total_entropy = -sum(probabilities .* log2(probabilities));
end
