module MyDecisionTree
    using DataFrames, StatsBase
export train!, DecisionTreeClassifier, RandomForestClassifier

function class_distribution(data, classes)
    d = Dict{Int64, Int64}()
    for class in classes
        d[class] = count(i->(i==class), data)
    end
    return d
end

#claculates gini impurity for a class distribution
function gini_impurity(dist)
    nums = [v for v in values(dist)]
    probs = (nums./sum(nums)).^2
    return 1-sum(probs)
end


abstract type Node end

mutable struct ClassifierNode <: Node
    left::Union{Node, Missing}
    right::Union{Node, Missing}
    split::Union{String, Missing}
    depth::Union{Int64, Missing}
    thresh::Union{Float64, Missing}
    gini::Union{Float64, Missing}
    class_count::Union{Dict{Int64, Int64}, Missing}
end

ClassifierNode() = ClassifierNode(missing, missing, missing, 0, missing, missing, missing)
ClassifierNode(i::Int64) = ClassifierNode(missing, missing, missing, i, missing, missing, missing)
mutable struct DecisionTreeClassifier
    root::ClassifierNode
    features::Union{Missing, Vector{String}}
    impurity_measure::Function
    max_depth::Int64
    min_samples_leaf::Int64
    min_impurity_decrease::Float64
end

DecisionTreeClassifier(features::Union{Missing, Vector{String}}, 
                        impurity_measure::Function = gini_impurity,  
                        max_depth::Int64=5, 
                        min_samples_leaf::Int64=20,min_impurity_decrease::Float64=1e-2) =  
DecisionTreeClassifier(ClassifierNode(), 
                        features,
                        impurity_measure, 
                        max_depth, min_samples_leaf,
                        min_impurity_decrease)

DecisionTreeClassifier(impurity_measure::Function = gini_impurity,  
                        max_depth::Int64=5, 
                        min_samples_leaf::Int64=20,min_impurity_decrease::Float64=1e-2) =  
DecisionTreeClassifier(ClassifierNode(), 
                        missing,
                        impurity_measure, 
                        max_depth, min_samples_leaf,
                        min_impurity_decrease)

mutable struct RandomForestClassifier
    n_classifiers::Int64
    n_samples::Int64
    n_sub_features::Int64
    trees::Vector{DecisionTreeClassifier}
    tree_constructor_args::Union{Missing, Dict{String, Any}}
end






RandomForestClassifier(n_classifiers::Int64, 
                        n_samples::Int64,n_sub_features::Int64, 
                        tree_constructor_args::Union{Missing, Dict{String, Any}}) = 

RandomForestClassifier(n_classifiers, 
                    n_samples,n_sub_features,
                    [DecisionTreeClassifier(tree_constructor_args["impurity_measure"][i],tree_constructor_args["max_depth"][i], tree_constructor_args["min_samples_leaf"][i], tree_constructor_args["min_impurity_decrease"][i]) for i in 1:n_classifiers], tree_constructor_args)


RandomForestClassifier(n_classifiers::Int64, 
                            n_samples::Int64,
                                n_sub_features::Int64) = 

RandomForestClassifier(n_classifiers, 
                    n_samples,n_sub_features,
                    [DecisionTreeClassifier(gini_impurity,5,20,1e-2) for i in 1:n_classifiers], missing)







function train!(classifier_node::ClassifierNode, 
                    data::DataFrame, target_name::Symbol, classes::Vector{Int64}, 
                        impurity_measure::Function ,maxdepth::Int64 = 5, 
                            min_samples_leaf::Int64 = 3, min_impurity_decrease::Float64 = 0.1)
    
    if classifier_node.depth >= maxdepth || size(data)[1] <= min_samples_leaf
        #turn node into leaf
        #classifier_node = Leaf(classifier_node)
        return
    end
    #initialization
    feature_names = [name for name in names(data) if name != String(target_name)]
    
    best_feature = ""
    best_split_val = Inf
    #initial distribution of classes for node
    init_dist = class_distribution(data[!, target_name], classes)

    #GINIs
    parent_gini = impurity_measure(init_dist)
    best_gini = impurity_measure(init_dist)

    
    #set for the current node
    classifier_node.class_count = copy(init_dist)
    classifier_node.gini = copy(best_gini)
    
    for feature in feature_names
            values = sort(data, [Symbol(feature)])
            #calculate the initial distributio
            splitvals = [quantile(values[!, Symbol(feature)], t) for t in 0.01:0.01:0.99]
            for splitval in splitvals
                prediction_mask = values[!, Symbol(feature)] .< splitval
                subset = data[prediction_mask, :]
                new_dist = class_distribution(subset[!, target_name], classes)
                curr_gini = impurity_measure(new_dist)            
                if (curr_gini <= best_gini) && (size(subset)[1] >= min_samples_leaf)
                    #println("New_best_gini:", curr_gini)
                    best_gini = copy(curr_gini)
                    best_split_val = copy(splitval)
                    best_feature = feature
                end
            end
        
    end
    
    if best_feature != ""

        # Update classifier_node with the best split information
       
                
        # Create left child node4
    
        if (abs(best_gini-parent_gini) >= min_impurity_decrease)
            
            classifier_node.split = best_feature
            classifier_node.thresh = best_split_val
           
            classifier_node.left = ClassifierNode(classifier_node.depth + 1)
            classifier_node.left.gini = copy(best_gini)
           
            left_mask = data[!, best_feature] .< best_split_val
            left_data = data[left_mask, :]
            classifier_node.left.class_count = class_distribution(left_data[!, target_name], classes)
            
            train!(classifier_node.left, left_data, target_name,classes, impurity_measure, maxdepth, min_samples_leaf)
        
            # Create right child node
            right_data = data[.!left_mask, :]
            right_dist = class_distribution(right_data[!, target_name], classes)
            
            right_gini =  impurity_measure(right_dist)
        
            classifier_node.right = ClassifierNode(classifier_node.depth + 1)            
            classifier_node.right.gini = copy(right_gini)
            classifier_node.right.class_count = right_dist
            train!(classifier_node.right, right_data, target_name,classes, impurity_measure, maxdepth, min_samples_leaf)
            
        else return end
    else return end
end

function train!(tree::DecisionTreeClassifier, data::DataFrame, target_name::Symbol, classes::Vector{Int64})
    train!(tree.root, data, target_name,classes, tree.impurity_measure, tree.max_depth, tree.min_samples_leaf, tree.min_impurity_decrease)
end




function bootstrap_subset(data::DataFrame, n_sub_features::Int64, n_samples::Int64=0, target_name::String="class")
    N = size(data)[1]
    feature_names = [name for name in names(data) if name != String(target_name)]
    
    if n_sub_features >= N
        prinln("Too much subfeatures defined, going to use N-1")
        n_sub_features = N -1
    end
    
    if n_samples == 0
        n_samples = N
    end
    subset_features = sample(feature_names, n_sub_features; replace=false)
    push!(subset_features, target_name)
    subset_idxs = rand(1:N, n_samples)
    subset = data[subset_idxs, subset_features]
    return subset
end

function train!(forest::RandomForestClassifier, dataset::DataFrame)
    for t in forest.trees
        subset = bootstrap_subset(dataset, 3, 0, "class")
        feature_names = [name for name in names(subset) if name != "class"]
        t.features = feature_names
        train!(t, subset, :class, [0, 1, 2])
    end
end


function calculate_prob(d)
    m = 0
    mk = 0
    for k in keys(d)
        if d[k]>m
            m = d[k]
            mk = k
        end
    end
    return mk
end

function predict_majority(RFC::RandomForestClassifier, x::Union{DataFrame, DataFrameRow}, cstr)
    prediction_counts = Dict{String, Int64}([(cls, 0) for cls in values(cstr)])
    for DT in RFC.trees
        x_sub = x[DT.features]
        curr_node = DT.root
        while !ismissing(curr_node.thresh)
            if x[curr_node.split] < curr_node.thresh
                curr_node = curr_node.left
            else
                curr_node = curr_node.right
            end
        end
        prediction_counts[cstr[calculate_prob(curr_node.class_count)]] += 1
    end
    return collect(keys(prediction_counts))[argmax(collect(values(prediction_counts)))]
end



#just a helper function to show the tree structure
#approximately....
function traverse(node::ClassifierNode, spacing, class_to_str)
    
    #operations on current node
    dp = 15
    p = repeat(" ", spacing)
    
    
    if !ismissing(node.thresh)
        println("$(p)|$(node.split) < $(node.thresh)")
        println("$(p)|class distribution: $(values(node.class_count))")
        println("$(p)-----------------------------------------------")
    else
        l = length("$(p)Predicted Class: $(  class_to_str[calculate_prob(node.class_count)])")
        #p2 = repeat(" ", spacing+12)
        
        println("$(p)Predicted Class: $(  class_to_str[calculate_prob(node.class_count)])")
        println("$(p)|class distribution: $(values(node.class_count))")
    end
    
    if !ismissing(node.left)
        for i in 1:5
            np = repeat(" ", spacing-i)
            pp = repeat(" ", spacing+2*i)
            if !ismissing(node.right)
               println("$(np)/$(pp)\\")
            else
               println("$(np)")
            end
                
        end
        traverse(node.left, spacing-dp, class_to_str)
    end
    if !ismissing(node.right)
         for i in 1:5
            np = repeat(" ", 2*spacing+i+8)
            println("$(np)\\")
        end
        traverse(node.right, 2*spacing-dp, class_to_str)
    end
end


function print_rft_tree(RFC::RandomForestClassifier, n::Int64)
    traverse(RFC.trees[n].root, 30, class_to_str)
end







end