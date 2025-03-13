clear
%% 导入数据
% 导入总奖牌数
total_num = readtable('total_counts.xlsx','ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 导入金牌数
golden_medal_num = readtable('Golden_medal_counts.xlsx','ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 导入东道主国家
host = readtable('summerOly_hosts.xlsx','ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 导入参加的人数
participants = readtable('participants.xlsx','ReadVariableNames', true, 'VariableNamingRule', 'preserve');

% 导入参加的项目
sports = readtable('sports_num.xlsx','ReadVariableNames', true, 'VariableNamingRule', 'preserve');
%% 输入参数
nation_name = "China";
nation_mini = "CHN";
serve_year = 12 ;

total_nation = total_num(total_num.NOC == nation_name, :);
golden_nation = golden_medal_num(golden_medal_num.NOC == nation_name, :);
participants_nation = participants(participants.NOC == nation_mini, :);
sports_nation = sports(sports.NOC == nation_mini, :);

total_nation = removevars(total_nation, total_nation.Properties.VariableNames{1});
golden_nation = removevars(golden_nation, golden_nation.Properties.VariableNames{1});
participants_nation = removevars(participants_nation, participants_nation.Properties.VariableNames{1});
sports_nation = removevars(sports_nation, sports_nation.Properties.VariableNames{1});
%处理后的数据
total_nation= table2array(total_nation);
golden_nation = table2array(golden_nation);
part_num= table2array(participants_nation);
sports_nation = table2array(sports_nation);
% 输出东道主标志
% 初始化一个长度为30的零数组
host_array = zeros(1, 30);
% 遍历从1904年到2024年，每4年一次
for year = 1904:4:2024
    % 查找是否该年份是东道主
    year_host = host(host.Year == year, :);
    
    % 如果该年是东道主
    if ~isempty(year_host) && year_host.Host{1} == nation_name
        % 计算该年份在数组中的位置
        year_index = (year - 1904) / 4 + 1;
        
        % 将对应位置设置为1
        host_array(year_index) = 8;
    end
end
area = zeros(1,30);
area(3) = 1;
%% 使用随机森林找重要性
input = [host_array',golden_nation',total_nation',part_num',sports_nation',area'];
feature_names = {'东道主', '金牌', '总奖牌', '参与人数', '参与项目','地域'};
feature_importance = measure_association_regression(input, total_nation, 500,feature_names);
