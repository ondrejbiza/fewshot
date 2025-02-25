

#generate possible part combinations for a given object to produce descriptors 
#for each of them test on n cases
#see what part combination works well for maximizing generality
#can also add one for inclusion or exclusion of 


part_names = []
all_possible_part_pairs = []
all_sets_of_all_possible_part_pairs = []
print(len(all_sets_of_all_possible_part_pairs))

#we should determine if the performance of relational descriptors is always additive
#on a complex teapot set probably 
#compare the reconstruction of indiv parts as you add more and more. 
#if they are universally additive test on harder teapots

#if they aren't universally additive see if some individually bad ones cause problems
#if it's combinations that can make things worse we can probably tree prune with that. just hill climb and prune anything worse than the best by a threshold
#then we can try also the pca/variational descriptors thing
#see if you can just do it to the training set or if you need particularly weird examples

#do this on partseg per category
#can use a few categories as training examples for better search heuristics generally? like something like training a prior on the bayesian program synthesis thing
#idk if i even care that much anymore but it could be fun if its not hard to implement
# could include some kind of like. weird latent space learned program type things to id similar kinds of parts of object that could be used similarly 