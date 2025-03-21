import React, { useState } from "react";
import { Link } from "react-router-dom";
import styles from "./NavigationBar.module.css";

function NavigationBar() {
  const [menuOpen, setMenuOpen] = useState(false);

  const toggleMenu = () => {
    setMenuOpen((prev) => !prev);
  };

  return (
    <nav className={styles.navBar}>
      <div className={styles.leftSide}>
        <img src="/logo.png" alt="SignIt Logo" className={styles.logo} />
        <span className={styles.title}>SignIt</span>
      </div>
      <div className={styles.rightSide}>
        <Link to="/">
          <img src="/home-icon.png" alt="Home" className={styles.homeIcon} />
        </Link>
        <div className={styles.menuContainer}>
          <img
            src="/menu-icon.png"
            alt="Menu"
            className={styles.menuIcon}
            onClick={toggleMenu}
          />
          {menuOpen && (
            <div className={styles.dropdownMenu}>
              <Link
                to="/asl-to-english"
                className={styles.dropdownItem}
                onClick={() => setMenuOpen(false)}
              >
                ASL → English
              </Link>
              <Link
                to="/english-to-asl"
                className={styles.dropdownItem}
                onClick={() => setMenuOpen(false)}
              >
                English → ASL
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}

export default NavigationBar;
