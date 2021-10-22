## Example notebooks for the implemented Collaborative Fitering methods.

* You may notice that the predicted rating of some (user, item) is larger than 5, although the largest rating in the data is 5. This is not a bug, but indeed the formula use to compute the predicted rating can produce a rating larger than 5! For example, consider the case we want to predict the rating of a user to a target item, where the formula of the prediction is: 

  ![image](https://user-images.githubusercontent.com/54271806/137902913-c63c2ecc-64f2-4f38-af1c-f7f314dcb0d0.png)


 with: user_mean_rating =  3.59, neighbor user mean rating = {user1 : 3.74, user2 : 2.7}, neighbors rating {user1: 5.0, user2: 5.0}, similarity_score = {user1:0.85, user2: 0.53}.
 
 Compute it yourself you'll see!
