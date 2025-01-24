using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

public class Button : Wall
{
    public bool IsPressed { get; private set; }
    public Button(Vector2 position, Vector2 size, String roll = "BUTTON") : base(position, size, roll) { }

    public void Press()
    {
        IsPressed = true;
    }

    public void Release()
    {
        IsPressed = false;
    }

    public new void Update(GameTime gameTime)
    {
        base.Update(gameTime);
    }

}