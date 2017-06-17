# Classical Boundary Trees
---
Hello! So, here's that iPython notebook I promised on Classical Boundary Trees.
Let's get started.

The Boundary Tree algorithm is an example of online, instance-based learning. What this means is, we can feed it independant examples, one-by-one, and have the algorithm learn as time goes on. The key for online learning of any kind is speed. The human brain has a response time of around 50-80ms, if any learning algorithm seeks to be able to replicate human ability - it's maximum upper bound on speed better be close to that. The Boundary Tree algorithm derives it's ability to be fast from its data structure, and of course what that data structure holds. Trees have an insert and retrieval time in O(log_b(n)), where b is the branching factor, the number of children each node is allowed to have.

So that covers that, initially. We'll be focusing only on classification - which is a supervised learning process; meaning that we provide a class label along with an example's feature vector. In general, the process is usually as follows:
    1.) Feed an unknown example's feature vector into our algorithm and see what class it "thinks" it is.
    2.) If the returned class matches the class label we said it was, we're all good; if not - some changes need to be made to fix the error. 
When those changes are made, it's said that we're training the algorithm - and as we all know, in order to learn, one must train. 

## Training and Querying
---
In order to train the Boundary Tree classifier, we need to start at the root. In order to "root" the tree, you would randomly sample a training example from your dataset, build a node (that can hold at least k children, depending on the language you use) from the feature vector and class label, and then as a special case insert - store it as the root of the tree.
From this point forth, it's quite a bit like k-Nearest Neighbor. If you know that algorithm, you can just scan the text following this and then start implementing. The Boundary Tree algorithm, like kNN, uses a distance metric to facilitate learning. We test a new example against those we've stored by taking the distance between them, returning the class of the closest, and checking if the actual class, and what the tree said it was, are equal. If they are not, *unlike* kNN (which just appends examples to an unordered collection), we add our new example and it's class as a child to the "closest" example we had. As a side note, feature vectors are usually floating-point numeric arrays or matrices, so "distances between examples" usually isn't as complicated as it sounds. 
(However, if the following makes sense to you - you can utilize it, if you feel the need: the distance function can be any functional that satisfies the properties of a metric space, so you're not limited to just Euclidean distance.)

In other words, we find the object in the tree that is closest (by whatever representation we choose) and if the classes between those two things don't match - since we've found the absolute closest the tree has to offer - we add a new child that is the example we've failed to identify. 

Algorithmically speaking, in order to find the closest node - we start with a "test" function the implementation should have, called **query**. The query function should be able to accept a test example's feature vector, and will do the following:
    1. Start by initializing a node transition variable to the tree's root.
    2. Retrieve all of the children that node has, and if the number of children is *less* than a defined maximum, *k*, grab the node itself, too (because there's still space to add a child; this is actually what our stopping condition, as well).
    3. Find the node in that collection which has the closest feature vector to our test example's. 
        3.1. If the node that is the closest ends up being the "parent" or, perhaps, leaf, for the level we're on - we've reached the best possible example we have and we return that node (which contains the feature vector and the class label).
        3.2. Otherwise, we set the node transition variable to the closest node and repeat from step 2.
    

**query** is the basis of the algorithm. It does all the work traversing the tree, and builds up the decision boundary implicitly. 

In order to train, there needs to be another function **train**. This function should accept an example's feature vector, and it's actual class label. **train** does the following, in our case of classification:
    1. Feed the test example's feature vector into **query**, store the node it returns. 
        1.1. If the returned node's class label **strictly does not equal** the test example's actual class, give the returned closest node a child that is a node composed of our new example. 
        1.2. Otherwise, don't do anything and just return the closest node's class. 

Train is really the only function you would ever need to call in practice, assuming you wanted the ability to continuously learn. It would really just return **query**'s response until you input an example the tree couldn't correctly classify. 

---
# Implementation

Okay, now we're going to implement the Boundary Tree algorithm. 
