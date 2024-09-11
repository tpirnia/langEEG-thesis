def permutation_test_score(
    estimator,
    X,
    y,
    *,
    groups=None,
    cv=None,
    n_permutations=100,
    n_jobs=None,
    random_state=0,
    verbose=0,
    scoring=None,
    fit_params=None,
):
    """Evaluate the significance of a cross-validated score with permutations
        Permutes targets to generate 'randomized data' and compute the empirical
        p-value against the null hypothesis that features and targets are
        independent.
        The p-value represents the fraction of randomized data sets where the
        estimator performed as well or better than in the original data. A small
        p-value suggests that there is a real dependency between features and
        targets which has been used by the estimator to give good predictions.
        A large p-value may be due to lack of real dependency between features
        and targets or the estimator was not able to use the dependency to
        give good predictions.
        Read more in the :ref:`User Guide <permutation_test_score>`.
        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.
        X : array-like of shape at least 2D
            The data to fit.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
            The target variable to try to predict in the case of
            supervised learning.
        groups : array-like of shape (n_samples,), default=None
            Labels to constrain permutation within groups, i.e. ``y`` values
            are permuted among samples with the same group identifier.
            When not specified, ``y`` values are permuted among all samples.
            When a grouped cross-validator is used, the group labels are
            also passed on to the ``split`` method of the cross-validator. The
            cross-validator uses them for grouping the samples  while splitting
            the dataset into train/test set.
        scoring : str or callable, default=None
            A single str (see :ref:`scoring_parameter`) or a callable
            (see :ref:`scoring`) to evaluate the predictions on the test set.
            If `None` the estimator's score method is used.
        cv : int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - `None`, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a `(Stratified)KFold`,
            - :term:`CV splitter`,
            - An iterable yielding (train, test) splits as arrays of indices.
            For `int`/`None` inputs, if the estimator is a classifier and `y` is
            either binary or multiclass, :class:`StratifiedKFold` is used. In all
            other cases, :class:`KFold` is used. These splitters are instantiated
            with `shuffle=False` so the splits will be the same across calls.
            Refer :ref:`User Guide <cross_validation>` for the various
            cross-validation strategies that can be used here.
            .. versionchanged:: 0.22
                `cv` default value if `None` changed from 3-fold to 5-fold.
        n_permutations : int, default=100
            Number of times to permute ``y``.
        n_jobs : int, default=None
            Number of jobs to run in parallel. Training the estimator and computing
            the cross-validated score are parallelized over the permutations.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        random_state : int, RandomState instance or None, default=0
            Pass an int for reproducible output for permutation of
            ``y`` values among samples. See :term:`Glossary <random_state>`.
        verbose : int, default=0
            The verbosity level.
        fit_params : dict, default=None
            Parameters to pass to the fit method of the estimator.
            .. versionadded:: 0.24
        Returns
        -------
        score : float
            The true score without permuting targets.
        permutation_scores : array of shape (n_permutations,)
            The scores obtained for each permutations.
        pvalue : float
            The p-value, which approximates the probability that the score would
            be obtained by chance. This is calculated as:
            `(C + 1) / (n_permutations + 1)`
            Where C is the number of permutations whose score >= the true score.
            The best possible p-value is 1/(n_permutations + 1), the worst is 1.0.
        Notes
        -----
        This function implements Test 1 in:
            Ojala and Garriga. `Permutation Tests for Studying Classifier
            Performance
            <http://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf>`_. The
            Journal of Machine Learning Research (2010) vol. 11
        """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    score = _permutation_test_score(
        clone(estimator), X, y, groups, cv, scorer, fit_params=fit_params
    )
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_permutation_test_score)(
            clone(estimator),
            X,
            _shuffle(y, groups, random_state),
            groups,
            cv,
            scorer,
            fit_params=fit_params,
        )
        for _ in range(n_permutations)
    )
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)
    return score, permutation_scores, pvalue


def _permutation_test_score(estimator, X, y, groups, cv, scorer, fit_params):
    """Auxiliary function for permutation_test_score"""
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    avg_score = []
    for train, test in cv.split(X, y, groups):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        fit_params = _check_fit_params(X, fit_params, train)
        estimator.fit(X_train, y_train, **fit_params)
        avg_score.append(scorer(estimator, X_test, y_test))
    return np.mean(avg_score)


def _shuffle(y, groups, random_state):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = groups == group
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return _safe_indexing(y, indices)