using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
public class Wall : GameObject
{

    public Wall(Vector2 position, Vector2 size, String roll = "WALL") : base(position, size, roll)
    {
        gravity = 0;
    }



}
