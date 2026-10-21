-- Creates a view need_meeting listing students with score < 80
-- and no last_meeting, or a last_meeting more than 1 month ago
CREATE OR REPLACE VIEW need_meeting AS
SELECT name
FROM students
WHERE score < 80
  AND (last_meeting IS NULL
       OR last_meeting < SUBDATE(CURDATE(), INTERVAL 1 MONTH));