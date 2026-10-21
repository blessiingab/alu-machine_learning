
DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;

DELIMITER $$

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(
    IN user_id INT
)
BEGIN
    UPDATE users
    SET average_score = (
        SELECT SUM(c.score * p.weight) / SUM(p.weight)
        FROM corrections AS c
        INNER JOIN projects AS p
            ON c.project_id = p.id
        WHERE c.user_id = user_id
    )
    WHERE id = user_id;
END $$

DELIMITER ;
