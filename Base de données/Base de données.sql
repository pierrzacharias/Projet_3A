-- phpMyAdmin SQL Dump
-- version 4.6.5.2
-- https://www.phpmyadmin.net/
--
-- Client :  localhost
-- Généré le :  Mer 03 Octobre 2018 à 13:22
-- Version du serveur :  5.7.23-0ubuntu0.16.04.1
-- Version de PHP :  7.0.32-0ubuntu0.16.04.1

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de donn�es :  VOYAGES
--

-- --------------------------------------------------------

--
-- Structure de la table `CLIENT`
--

CREATE TABLE `CLIENT` (
  numC char(5) NOT NULL,
  nom varchar(100) NOT NULL,
  prénom varchar(100) NOT NULL,
  adresse varchar(200) DEFAULT NULL,
  ville varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Contenu de la table `CLIENT`
--

INSERT INTO `CLIENT` (numC, nom, prénom, adresse, ville) VALUES
('C0001', 'Piroux', 'Samy', '12, rue de la colline  54000', 'NANCY'),
('C0002', 'Fanier', 'Gabriel', '43, rue des lilas 69000', 'LYON'),
('C0003', 'Vigor', 'Denis', '65, rue de la convention 54520', 'LAXOU'),
('C0004', 'Duponoy', 'Guillaume', '234, rue Emile Zola 59000', 'LILLE'),
('C0005', 'Abel', 'Claire', '34, rue sous la croix 54000', 'NANCY'),
('C0006', 'Finiet', 'Lise', '123, rue Scrapone 54200', 'TOUL'),
('C0007', 'Michalo', 'Charlotte', '74, rue des accacias 88000', 'EPINAL'),
('C0008', 'Ribeira', 'Madeleine', '85, rue Bartoldi 45100', 'METZ'),
('C0009', 'Rouva', 'Héléna', '153, rue de la reine 55110', 'VAUCOULEURS'),
('C0010', 'Boulard', 'Laure', '232, bd des armées 57000', 'METZ'),
('C0011', 'Bernard', 'Alix', '2bis, rue de la pie qui chante 51100', 'REIMS'),
('C0012', 'Jeandel', 'Sacha', '59, rue de Haribo 51100', 'REIMS');

-- --------------------------------------------------------

--
-- Structure de la table RESERVATION
--

CREATE TABLE RESERVATION (
  numC char(5) NOT NULL DEFAULT '',
  codeV char(6) NOT NULL,
  dateRes date NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Contenu de la table RESERVATION
--

INSERT INTO RESERVATION (numC, codeV, dateRes) VALUES
('C0001', 'V00110', '2015-01-01'),
('C0002', 'V00120', '2005-11-30'),
('C0003', 'V00100', '2014-08-05'),
('C0003', 'V00110', '2015-01-11'),
('C0003', 'V00120', '2005-11-30'),
('C0003', 'V00130', '2015-09-30'),
('C0003', 'V00140', '2014-08-05'),
('C0004', 'V00120', '2005-09-30'),
('C0005', 'V00110', '2015-11-30'),
('C0005', 'V00130', '2015-09-17'),
('C0006', 'V00120', '2005-09-30'),
('C0006', 'V00140', '2014-09-20'),
('C0008', 'V00100', '2014-08-10'),
('C0008', 'V00120', '2005-11-12'),
('C0009', 'V00130', '2015-09-15'),
('C0010', 'V00120', '1999-11-30');

-- --------------------------------------------------------

--
-- Structure de la table VOYAGE
--

CREATE TABLE VOYAGE (
  codeV char(6) NOT NULL,
  villeDestination varchar(100) NOT NULL,
  dateV date NOT NULL,
  durée decimal(6,2) DEFAULT NULL,
  prixBillet decimal(6,2) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Contenu de la table VOYAGE
--

INSERT INTO VOYAGE (codeV, villeDestination, dateV, durée, prixBillet) VALUES
('V00100', 'ROME', '2014-12-01', '20.00', '580.00'),
('V00110', 'DIJON', '2015-02-01', '3.50', '288.50'),
('V00120', 'NICE', '2005-12-01', '10.00', '280.00'),
('V00130', 'AVIGNON', '2015-12-01', '6.00', '290.00'),
('V00140', 'METZ', '2014-12-01', '1.50', '400.00'),
('V00150', 'NICE', '2016-01-07', '13.00', '120.00'),
('V00160', 'BRUXELLES', '2015-12-22', '5.00', '70.00');

--
-- Index pour les tables exportées
--

--
-- Index pour la table `CLIENT`
--
ALTER TABLE `CLIENT`
  ADD PRIMARY KEY (numC);

--
-- Index pour la table RESERVATION
--
ALTER TABLE RESERVATION
  ADD PRIMARY KEY (numC,codeV),
  ADD KEY fk_reservation_voyage (codeV);

--
-- Index pour la table VOYAGE
--
ALTER TABLE VOYAGE
  ADD PRIMARY KEY (codeV);

--
-- Contraintes pour les tables exportées
--

--
-- Contraintes pour la table RESERVATION
--
ALTER TABLE RESERVATION
  ADD CONSTRAINT fk_reservation_client FOREIGN KEY (numC) REFERENCES CLIENT (numC),
  ADD CONSTRAINT fk_reservation_voyage FOREIGN KEY (codeV) REFERENCES VOYAGE (codeV);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
