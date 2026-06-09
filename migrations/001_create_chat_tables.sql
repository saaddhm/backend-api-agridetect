-- ===========================================================================
-- AgriDetect AI — Migration : tables du chat de support (MySQL 8+)
-- ---------------------------------------------------------------------------
-- A exécuter une seule fois sur la base de production.
-- En développement (SQLite/MySQL), SQLModel.metadata.create_all() crée déjà
-- ces tables automatiquement au démarrage ; ce script sert pour un déploiement
-- maîtrisé / versionné des schémas en production.
-- ===========================================================================

SET NAMES utf8mb4;

-- --------------------------------------------------------------------------
-- conversation
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `conversation` (
  `id`                INT NOT NULL AUTO_INCREMENT,
  `user_id`           INT NOT NULL,
  `assigned_admin_id` INT NULL,
  `status`            VARCHAR(16) NOT NULL DEFAULT 'open',
  `created_at`        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_message_at`   DATETIME NULL,
  PRIMARY KEY (`id`),
  KEY `ix_conversation_user_id` (`user_id`),
  KEY `ix_conversation_assigned_admin_id` (`assigned_admin_id`),
  KEY `ix_conversation_status` (`status`),
  KEY `ix_conversation_last_message_at` (`last_message_at`),
  CONSTRAINT `fk_conversation_user`
    FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_conversation_admin`
    FOREIGN KEY (`assigned_admin_id`) REFERENCES `user` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- message
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `message` (
  `id`              INT NOT NULL AUTO_INCREMENT,
  `conversation_id` INT NOT NULL,
  `sender_id`       INT NOT NULL,
  `sender_role`     VARCHAR(8) NOT NULL,           -- 'user' | 'admin'
  `content`         TEXT NOT NULL,
  `is_read`         TINYINT(1) NOT NULL DEFAULT 0,
  `created_at`      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `ix_message_conversation_id` (`conversation_id`),
  KEY `ix_message_is_read` (`is_read`),
  KEY `ix_message_created_at` (`created_at`),
  CONSTRAINT `fk_message_conversation`
    FOREIGN KEY (`conversation_id`) REFERENCES `conversation` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_message_sender`
    FOREIGN KEY (`sender_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- --------------------------------------------------------------------------
-- devicetoken (ciblage des notifications push FCM)
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `devicetoken` (
  `id`         INT NOT NULL AUTO_INCREMENT,
  `user_id`    INT NOT NULL,
  `token`      VARCHAR(512) NOT NULL,
  `platform`   VARCHAR(32) NULL,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_devicetoken_token` (`token`),
  KEY `ix_devicetoken_user_id` (`user_id`),
  CONSTRAINT `fk_devicetoken_user`
    FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
