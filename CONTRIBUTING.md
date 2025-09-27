# 1. Look through the code and try to follow the existing conventions

  The reason for this requirement is to make the code look familiar to everyone.

  Don't give the code to AI, without explicitly asking it to follow the conventions.

# 2. Test your proposed changes as much as you can

   The reason for this requirement is to try to keep the software functional.

   If possible, perform manual testing in addition to executing the automatic tests, described in the [README](./README.md#running-tests).

# 3. Use useful comments in the code

   The reason for this requirement is to make the software easier to understand.

   Avoid uninformative, vacuous comments, such as those that tend to be written by AI.
   Add comments when the code is hard to understand, or to explain edge cases.
   See more discussion about why comments are useful in [comments](https://github.com/johnousterhout/aposd-vs-clean-code?tab=readme-ov-file#comments).

# 4. Use imperative mood for the pull request title, and begin it with a capital letter

   The reason for this requirement is to have a convention, and avoid the situation shown in [https://xkcd.com/1296](https://xkcd.com/1296).

   ![https://xkcd.com/1296](https://imgs.xkcd.com/comics/git_commit.png)

   This means the title of the pull request is expected to complete the sentence:

   > When applied, this change will ...

   For example, this is an expected title format:

   - Implement authentication

   These are not:

   - Implementing authentication (it's not in the imperative mood)
   - implement authentication (starts with lower case)
   - Implemented authentication (it's not in the imperative mood)

# 5. Write the pull request description as a human, and explain the motivation for the change

   The reason for this requirement is to reduce the time needed to review the correctness of AI-generated text.

   Even if you use AI to help generate the PR, remove the AI-generated description and replace it with your own words.
   AI tends to generate a lot of text, and duplicated, inconsistent content.
   It also tends to use vacuous expressions, such as “smart changes,” “comprehensive documentation,” “streamlined implementation,” etc.

